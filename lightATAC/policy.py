import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, use_tanh=False):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.use_tanh = use_tanh

    def forward(self, obs, ignore_tanh=None):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        dist = MultivariateNormal(mean, scale_tril=scale_tril)
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