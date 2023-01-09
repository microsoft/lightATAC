
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from copy import deepcopy
from lightATAC.util import compute_batched, discount_cumsum, sample_batch, \
        traj_data_to_qlearning_data, tuple_to_traj_data, update_exponential_moving_average, normalized_sum, asymmetric_l2_loss

class BehaviorPretraining(nn.Module):
    """
        A generic pretraining algorithm for learning the behavior policy and its
        Q, value functions.

        It trains the policy by behavior cloning (MLE or L2 error), the Q
        function can be trained by TD error (with a target network) and/or
        residual error (using a double Q function), and the value function is
        trained by TD-lambda and expectile regression (by default, it uses least
        squares.)

    """

    def __init__(self, *,
                 policy=None,  # nn.module
                 qf=None,  # nn.module
                 vf=None,  # nn.module
                 discount=0.99,  # discount factor
                 lr=5e-4,  # learning rate
                 use_cache=False,
                 # Q learning
                 target_update_rate=5e-3,
                 td_weight=1.0,  # weight on the td error (surrogate based on target network)
                 rs_weight=0.0,  # weight on the residual error
                 # V learning
                 lambd=1.0,  # lambda for TD lambda
                 expectile=0.5,
                 # policy entropy
                 action_shape=None,
                 fixed_alpha=0,
                 target_entropy=None,
                 initial_log_alpha=0.,
                 ):
        """
            Inputs:
                policy: An nn.module of policy network that returns a distribution or a tensor.
                qf: An nn.module of double Q networks that implement additionally `both` method.
                vf: An nn.module of value network.

                Any provided networks above would be trained; to train `qf`, `policy` is required.

                discount: discount factor
                lr: learning rate
                use_cache: whether to batch compute the policy and cache the result
                target_update_rate: learning rate to update target network
                td_weight: weight on the TD loss for learning `qf`
                rs_weight: weight on the residual loss for learning `qf`
                lambd: lambda in TD-lambda for learning `vf`.
                expectile: expectile used for learning `vf`.

                action_shape: shape of the vector action space
                fixed_alpha: weight on the entropy term
                target_entropy: entropy lower bound; if None, it would be inferred from `action_shape`
                initial_log_alpha: initial log entropy
        """

        super().__init__()
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.discount = discount
        self.lambd = lambd
        self.td_weight = td_weight / (td_weight+rs_weight)
        self.rs_weight = rs_weight / (td_weight+rs_weight)
        self.target_update_rate = target_update_rate
        self.expectile = expectile

        if self.qf is not None:
            assert self.policy is not None, 'Learning a q network requires a policy network.'
            self.target_qf = deepcopy(self.qf).requires_grad_(False)

        if self.vf is not None:
            self.target_vf = deepcopy(self.vf).requires_grad_(False)

        parameters = []
        for x in (policy, qf, vf):
            if x is not None:
                parameters+= list(x.parameters())

        self.optimizer = torch.optim.Adam(parameters, lr=lr)

        # Cache
        self.use_cache = use_cache
        self._next_policy_outs = None

        # Control of policy entropy
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            assert action_shape is not None
            self._target_entropy = target_entropy if target_entropy else -np.prod(action_shape).item()
            self._log_alpha = torch.nn.Parameter(torch.tensor(initial_log_alpha))
            self._alpha_optimizer = torch.optim.Adam([self._log_alpha], lr=lr)
        else:
            self._log_alpha = torch.tensor(self._fixed_alpha).log()

    def train(self, dataset, n_steps, batch_size=256, log_freq=1000, log_fun=None, silence=False):
        """ A basic trainer loop. Users cand customize this method if needed.

            dataset: a dict of observations, actions, rewards, terminals
        """
        traj_data = tuple_to_traj_data(dataset)
        if self.vf is not None:
            self.preprocess_traj_data(traj_data, self.discount)
        dataset = traj_data_to_qlearning_data(traj_data)  # make sure `next_observations` is there

        for step in trange(n_steps, disable=silence):
            train_metrics = self.update(**sample_batch(dataset, batch_size))
            if (step % max(log_freq,1) == 0 or step==n_steps-1) and log_fun is not None:
                log_fun(train_metrics, step)
        return dataset

    def compute_qf_loss(self, observations, actions, next_observations, rewards, terminals, **kwargs):
        # Q Loss with TD error and/or Residual Error using Target Q
        def compute_bellman_backup(v_next):
            assert rewards.shape == v_next.shape
            return rewards + (1.-terminals.float())*self.discount*v_next
        qf_loss = 0.
        # Update target
        update_exponential_moving_average(self.target_qf, self.qf, self.target_update_rate)

        # Compute shared parts
        # We assured self.policy is not None, so self._next_policy_outs has been precomputed.
        next_policy_outs = self._next_policy_outs if self.use_cache else self.policy(next_observations)
        next_policy_actions = next_policy_outs.sample() if isinstance(next_policy_outs, torch.distributions.Distribution) else next_policy_outs
        next_policy_actions = next_policy_actions.detach()

        if self.rs_weight>0:
            qs_all, next_qs_all = compute_batched(self.qf.both, [observations,  next_observations],
                                                                [actions,       next_policy_actions])
        else:
            qs_all = self.qf.both(observations, actions)  # tuple, inference

        # Temporal difference error
        if self.td_weight>0:
            next_targets = self.target_qf(next_observations, next_policy_actions)  # inference
            td_targets = compute_bellman_backup(next_targets)
            for qs in qs_all:
                qf_loss += asymmetric_l2_loss(td_targets - qs, self.expectile) * self.td_weight
        # Residual error
        if self.rs_weight>0:
            # next_qs_all = self.qf.both(next_observations, next_policy_actions)  # inference
            for qs, next_qs in zip(qs_all, next_qs_all):
                rs_targets = compute_bellman_backup(next_qs)
                qf_loss += asymmetric_l2_loss(rs_targets - qs, self.expectile) * self.rs_weight
        # Log
        info_dict = {"Q loss": qf_loss.item(),
                     "Average Q value": qs.mean().item()}
        return qf_loss, info_dict

    def compute_vf_loss(self, observations, actions, next_observations, rewards, terminals,
                      returns, remaining_steps, last_observations, last_terminals, **kwargs):
        # V loss (TD-lambda)

        # Monte-Carlo Q estimate
        mc_estimates = torch.zeros_like(rewards)
        update_exponential_moving_average(self.target_vf, self.vf, self.target_update_rate)
        if self.lambd>0:
            last_vs = self.target_vf(last_observations)  # inference
            mc_estimates = returns + (1-last_terminals.float()) * self.discount**remaining_steps * last_vs
        if self.lambd<1:
            with torch.no_grad():
                v_next = self.target_vf(next_observations)  # inference
                td_targets = rewards + (1.-terminals.float())*self.discount*v_next
        vs = self.vf(observations)  # inference
        # TD error
        td_error = 0.
        vf_loss = asymmetric_l2_loss(td_targets - vs, self.expectile)
        # Log
        info_dict = {"V loss": vf_loss.item(),
                     "Average V value": vs.mean().item()}
        return vf_loss, info_dict

    def compute_policy_loss(self, observations, actions, next_observations, rewards, terminals, **kwargs):
        # Policy loss

        if self.qf is not None and self.use_cache:
            policy_outs, self._next_policy_outs = compute_batched(self.policy, [observations, next_observations])
        else:
            policy_outs = self.policy(observations)

        if isinstance(policy_outs, torch.distributions.Distribution):  # MLE
            policy_loss = -policy_outs.log_prob(actions).mean()
            if self._log_alpha > -float("Inf"):
                new_actions_dist = policy_outs
                new_actions = new_actions_dist.rsample()
                log_pi_new_actions = new_actions_dist.log_prob(new_actions)
                policy_entropy = -log_pi_new_actions.mean()
                if self._use_automatic_entropy_tuning:
                    alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
                    self._alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self._alpha_optimizer.step()
                alpha = self._log_alpha.exp().detach()
                policy_loss = normalized_sum(policy_loss, -policy_entropy, alpha)

        elif torch.is_tensor(policy_outs):  # l2 loss
            assert policy_outs.shape == actions.shape
            policy_loss = torch.sum((policy_outs - actions)**2, dim=1).mean()
        else:
            raise NotImplementedError
        info_dict = {"Policy loss": policy_loss.item()}
        return policy_loss, info_dict

    def update(self, **batch):
        qf_loss = vf_loss = policy_loss = torch.tensor(0., device=batch['observations'].device)
        qf_info_dict = vf_info_dict = policy_info_dict = {}
        # Compute loss
        if self.policy is not None:
            policy_loss, policy_info_dict = self.compute_policy_loss(**batch)
        if self.qf is not None:
            qf_loss, qf_info_dict = self.compute_qf_loss(**batch)
        if self.vf is not None:
            vf_loss, vf_info_dict = self.compute_vf_loss(**batch)
        # Update
        loss = policy_loss + qf_loss + vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Log
        info_dict = {**qf_info_dict, **vf_info_dict, **policy_info_dict}
        return info_dict

    @classmethod
    def preprocess_traj_data(cls, traj_data, discount):
        for traj in traj_data:
            H = len(traj['rewards'])
            if torch.is_tensor(traj['rewards']):
                with torch.no_grad():
                    traj['returns'] = discount_cumsum(traj['rewards'], discount)
                    assert traj['returns'].shape == traj['rewards'].shape
                    traj['remaining_steps'] = torch.flip(torch.arange(H, device=traj['rewards'].device), dims=(0,))+1
                    assert traj['remaining_steps'].shape == traj['rewards'].shape
                    traj['last_observations'] = torch.repeat_interleave(traj['observations'][-1:], H, dim=0)
                    assert traj['last_observations'].shape ==traj['observations'].shape
                    traj['last_terminals'] = torch.repeat_interleave(traj['terminals'][-1], H)
                    assert traj['last_terminals'].shape == traj['terminals'].shape
            else:
                traj['returns'] = discount_cumsum(traj['rewards'], discount)
                assert traj['returns'].shape == traj['rewards'].shape
                traj['remaining_steps'] = np.flip(np.arange(H))+1
                assert traj['remaining_steps'].shape == traj['rewards'].shape
                traj['last_observations'] = np.repeat(traj['observations'][-1:], H, axis=0)
                assert traj['last_observations'].shape ==traj['observations'].shape
                traj['last_terminals'] = np.repeat(traj['terminals'][-1], H)
                assert traj['last_terminals'].shape == traj['terminals'].shape
