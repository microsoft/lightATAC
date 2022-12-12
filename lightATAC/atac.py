# yapf: disable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightATAC.util import compute_batched, DEFAULT_DEVICE, update_exponential_moving_average


def normalized_sum(loss, reg, w):
    return loss/w + reg if w>1 else loss + w*reg

def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

class ATAC(nn.Module):
    """ Adversarilly Trained Actor Critic """
    def __init__(self, *,
                 policy,
                 qf,
                 optimizer,
                 discount=0.99,
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 action_shape,  # shape of the action space
                 # Optimization parameters
                 policy_lr=5e-7,
                 qf_lr=5e-4,
                 target_update_tau=5e-3,
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 # ATAC parameters
                 beta=1.0,  # the regularization coefficient in front of the Bellman error
                 norm_constraint=100,  # l2 norm constraint on the NN weight
                 # ATAC0 parameters
                 init_observations=None, # Provide it to use ATAC0 (None or np.ndarray)
                 buffer_batch_size=256,  # for ATAC0 (sampling batch_size of init_observations)
                 # Misc
                 debug=True,
                 # Heuristic
                 heuristic_method='None',
                 heuristic_temperature=0.01,
                 ):

        #############################################################################################
        super().__init__()
        assert beta>=0 and norm_constraint>=0
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        self._debug = debug  # log extra info

        # ATAC main parameter
        self.beta = beta # regularization constant on the Bellman surrogate

        # q update parameters
        self._discount = discount
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.

        # networks
        self.policy = policy
        self._qf = qf
        self._target_qf = copy.deepcopy(self._qf).requires_grad_(False)

        # optimization
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._tau = target_update_tau

        self._optimizer = optimizer
        self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr) #  lr for warmstart
        self._qf_optimizer = self._optimizer(self._qf.parameters(), lr=self._qf_lr)

        # control of policy entropy
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            self._target_entropy = target_entropy if target_entropy else -np.prod(action_shape).item()
            self._log_alpha = torch.nn.Parameter(torch.Tensor([initial_log_entropy])) # torch.Tensor([self._initial_log_entropy]).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._alpha_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()

        # initial state pessimism (ATAC0)
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._buffer_batch_size = buffer_batch_size

        # heuristic
        self._heuristic_method = heuristic_method
        self._heuristic_temperature = heuristic_temperature

    def update(self, observations, actions, next_observations, rewards, terminals, **kwargs):

        rewards = rewards.flatten()
        terminals = terminals.flatten().float()

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return rewards + (1.-terminals)*self._discount*q_pred_next

        # Pre-computation
        with torch.no_grad():  # regression target
            new_next_actions = self.policy(next_observations).rsample()
            target_q_values = torch.clip(self._target_qf(next_observations, new_next_actions), min=self._Vmin, max=self._Vmax)  # projection
            q_target = compute_bellman_backup(target_q_values.flatten())

        # qf_pred_both = self._qf.both(observations, actions)
        # qf_pred_next_both = self._qf.both(next_observations, new_next_actions)
        new_actions_dist = self.policy(observations)  # This will be used to compute the entropy
        new_actions = new_actions_dist.rsample() # These samples will be used for the actor update too, so they need to be traced.

        if self._init_observations is None:  # ATAC
            pess_new_actions = new_actions.detach()
            pess_observations = observations
        else:  # initial state pessimism
            idx_ = np.random.choice(len(self._init_observations), self._buffer_batch_size)
            init_observations = self._init_observations[idx_]
            init_actions_dist = self.policy(init_observations)[0]
            pess_new_actions = init_actions_dist.rsample().detach()
            pess_observations = init_observations

        qf_pred_both, qf_pred_next_both, qf_new_actions_both \
            = compute_batched(self._qf.both, [observations, next_observations, pess_observations],
                                             [actions,      new_next_actions,  pess_new_actions])

        qf_loss = 0
        w1, w2 = 0.5, 0.5
        for qfp, qfpn, qfna in zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both):
            # Compute Bellman error
            assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape
            target_error = F.mse_loss(qfp, q_target)
            q_target_pred = compute_bellman_backup(qfpn)
            td_error = F.mse_loss(qfp, q_target_pred)
            qf_bellman_loss = w1*target_error+ w2*td_error
            # Compute pessimism term
            if self._init_observations is None:  # ATAC
                pess_loss = (qfna - qfp).mean()
            else:  # initial state pess. ATAC0
                pess_loss = qfna.mean()
            ## Compute full q loss (qf_loss = pess_loss + beta * qf_bellman_loss)
            qf_loss += normalized_sum(pess_loss, qf_bellman_loss, self.beta)

        # Update q
        self._qf_optimizer.zero_grad()
        qf_loss.backward()
        self._qf_optimizer.step()
        self._qf.apply(l2_projection(self._norm_constraint))
        update_exponential_moving_average(self._target_qf, self._qf, self._tau)

        ##### Update Actor #####
        # Compuate entropy
        log_pi_new_actions = new_actions_dist.log_prob(new_actions)
        policy_entropy = -log_pi_new_actions.mean()

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # Compute performance difference lower bound (policy_loss = - lower_bound - alpha * policy_kl)
        alpha = self._log_alpha.exp().detach()
        self._qf.requires_grad_(False)
        lower_bound = self._qf.both(observations, new_actions)[-1].mean() # just use one network
        self._qf.requires_grad_(True)
        policy_loss = normalized_sum(-lower_bound, -policy_entropy, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Log
        log_info = dict(policy_loss=policy_loss.item(),
                        qf_loss=qf_loss.item(),
                        qf_bellman_loss=qf_bellman_loss.item(),
                        pess_loss=pess_loss.item(),
                        alpha_loss=alpha_loss.item(),
                        policy_entropy=policy_entropy.item(),
                        alpha=alpha.item(),
                        lower_bound=lower_bound.item(),
        )

        # For logging
        if self._debug:
            with torch.no_grad():
                debug_log_info = dict(
                        bellman_surrogate=td_error.item(),
                        qf1_pred_mean=qf_pred_both[0].mean().item(),
                        qf2_pred_mean = qf_pred_both[1].mean().item(),
                        q_target_mean = q_target.mean().item(),
                        target_q_values_mean = target_q_values.mean().item(),
                        qf1_new_actions_mean = qf_new_actions_both[0].mean().item(),
                        qf2_new_actions_mean = qf_new_actions_both[1].mean().item(),
                        action_diff = torch.mean(torch.norm(actions - new_actions, dim=1)).item()
                        )
            log_info.update(debug_log_info)
        return log_info
