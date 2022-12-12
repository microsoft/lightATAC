
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from lightATAC.util import compute_batched, discount_cumsum, sample_batch, traj_to_tuple_data


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class BehaviorPretraining(nn.Module):
    """
        A generic pretraining algorithm for learning the behavior policy and its values.

        It trains the policy by behavior cloning (MLE or L2 error), and the
        values (v and q) by TD-lambda and expectile regression (by default, it
        uses least squares.)

    """

    def __init__(self, networks,  optimizers, discount=0.99, lambd=1.0, expectile=0.5):
        super().__init__()
        self.networks = networks
        self.optimizers = optimizers
        self.discount = discount
        self.lambd = lambd
        self.expectile = expectile

    def inference(self, observations, actions, next_observations, last_observations):
        """ Users can customize this method for their network architecture. """
        policy_dists = self.networks['policy'](observations)
        qs = self.networks['qf'](observations,actions)
        vs, next_vs, last_vs = compute_batched(self.networks['vf'], [observations, next_observations, last_observations])
        return policy_dists, qs, vs, next_vs, last_vs

    def train(self, traj_data, n_steps, batch_size=256, log_freq=1000, log_fun=None):
        """ A basic trainer loop. Users cand customize this method if needed.

            traj_data: a list of trajectory dicts
        """
        self.preprocess_traj_data(traj_data, self.discount)
        data = traj_to_tuple_data(traj_data)
        for step in trange(n_steps):
            train_metrics = self.update(**sample_batch(data, batch_size))
            if (step+1) % max(log_freq,1) == 0 and log_fun is not None:
                log_fun(train_metrics)
        return traj_data

    def update(self, observations, actions, next_observations, rewards, terminals,
                     returns, remaining_steps, last_observations, last_terminals, **kwargs):
        # Inference
        policy_dists, qs, vs, next_vs, last_vs = self.inference(observations, actions, next_observations, last_observations)

        # Value loss
        mc_estimate = returns + (1-last_terminals.float()) * self.discount**remaining_steps * last_vs
        td_estiamte = rewards + (1-terminals.float()) * self.discount * next_vs
        targets = (self.lambd * mc_estimate + (1-self.lambd) * td_estiamte).detach()
        v_loss = asymmetric_l2_loss(targets - vs, self.expectile)
        q_loss = asymmetric_l2_loss(targets - qs, self.expectile)

        # Policy loss
        if isinstance(policy_dists, torch.distributions.Distribution):  # MLE
            bc_losses = -policy_dists.log_prob(actions)
        elif torch.is_tensor(policy_dists):  # l2 loss
            assert policy_dists.shape == actions.shape
            bc_losses = torch.sum((policy_dists - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(bc_losses)

        # Update
        loss = policy_loss + q_loss + v_loss
        for opt in self.optimizers: opt.zero_grad()
        loss.backward()
        for opt in self.optimizers: opt.step()

        info_dict = {
            "V loss": v_loss.item(),
            "Q loss": q_loss.item(),
            "Policy loss": policy_loss.item(),
            "Average Q value": qs.mean().item(),
            "Average value value": vs.mean().item(),
            "Average target value": targets.mean().item()
        }

        return info_dict

    @classmethod
    def preprocess_traj_data(cls, traj_data, discount):
        for traj in traj_data:
            H = len(traj['rewards'])
            traj['returns'] = discount_cumsum(traj['rewards'], discount)
            assert traj['returns'].shape == traj['rewards'].shape
            traj['remaining_steps'] = np.flip(np.arange(H))+1
            assert traj['remaining_steps'].shape == traj['rewards'].shape
            traj['last_observations'] = np.repeat(traj['observations'][-1:], H, axis=0)
            assert traj['last_observations'].shape ==traj['observations'].shape
            traj['last_terminals'] = np.repeat(traj['terminals'][-1], H)
            assert traj['last_terminals'].shape == traj['terminals'].shape



class BehaviorCloning(BehaviorPretraining):

    def __init__(self, policy,  optimizer):
        super().__init__(policy, [optimizer])

    def inference(self, observations, actions, next_observations, last_observations):
        """ Users can customize this method for their network architecture. """
        policy_dists = self.networks(observations)
        qs = vs = next_vs = last_vs = torch.zeros(observations.shape[0], device=observations.device)
        return policy_dists, qs, vs, next_vs, last_vs