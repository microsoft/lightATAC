import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, TransformedDistribution, Normal
from torch.distributions.transforms import TanhTransform
import numpy as np
from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# Below are modified from gwthomas/IQL-PyTorch

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2,
                init_std=1.0, use_tanh=False, min_std=1e-5, max_std=10, std_type='constant'):
        super().__init__()
        init_log_std = np.log(init_std)
        self.std_type = std_type
        if self.std_type=='diagonal':
            # the first half of output predicts the mean; the second half predicts log_std
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim*2])
            self.net[-1].weight.data[act_dim:] *= 0.
            self.net[-1].bias.data[act_dim:] = init_log_std
        elif self.std_type=='constant':
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
            self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32)* init_log_std)
        else:
            raise ValueError
        self.use_tanh = use_tanh
        self.min_std = min_std
        self.max_std = max_std

    def forward(self, obs, ignore_tanh=None):
        if self.std_type=='diagonal':
            out = self.net(obs)
            mean, log_std = out.split(out.shape[-1]//2, dim=-1)
            std = torch.exp(log_std).clamp(self.min_std, self.max_std)
            dist = Normal(mean, std)
        elif self.std_type=='constant':
            mean = self.net(obs)
            std = torch.exp(self.log_std).clamp(self.min_std, self.max_std)
            scale_tril = torch.diag(std)
            dist = MultivariateNormal(mean, scale_tril=scale_tril)
        else:
            raise ValueError
        if self.use_tanh and not ignore_tanh:
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
        return dist

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs, ignore_tanh=True)  # just Gaussian
            act = dist.mean if deterministic else dist.sample()
            if self.use_tanh:
                act = torch.tanh(act)
            return act


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, use_tanh=False):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)
        self.use_tanh = use_tanh

    def forward(self, obs):
        return torch.tanh(self.net(obs)) if self.use_tanh else self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)