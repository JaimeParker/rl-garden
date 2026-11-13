"""A downstream package's own actor heads and feature extractor, trained by
rl-garden's PPO and SAC without subclassing an algorithm (the contracts are
in ``rl_garden/policies/contracts.py``).

- ``make_tanh_gaussian_ppo_actor``: an ``actor_factory`` for PPO. The head is
  a tanh-squashed Gaussian, which has no closed-form entropy, so it reports
  ``None``; its log-probability factors per action dimension.
- ``make_squashed_sac_actor``: an ``actor_factory`` for the SAC family, built
  on ``SACActorBase`` (two methods; the n-sample, external log-prob and
  diagnostics methods keep the base defaults).
- ``StateBackboneExtractor``: a ``BaseFeaturesExtractor`` for
  ``--actor-extractor``, standing in for a pretrained backbone; its features
  feed the default actor and critic heads.

On the CLI: ``--actor-factory rlg_downstream_example.byo_policy:make_tanh_gaussian_ppo_actor``,
``--actor-extractor rlg_downstream_example.byo_policy:StateBackboneExtractor``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import SACActorBase

#: Keeps ``atanh`` of a stored action at the bounds finite.
_ATANH_EPS = 1e-6
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0


def _bounds(action_space: spaces.Box) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.as_tensor(action_space.low, dtype=torch.float32).reshape(-1)
    high = torch.as_tensor(action_space.high, dtype=torch.float32).reshape(-1)
    return low, high


class TanhGaussianPPOActor(nn.Module):
    """A ``PPOActor``: ``a = bias + scale * tanh(u)``, ``u ~ N(mean(f), std)``.

    PPO scores the STORED action again during the update, so
    ``evaluate_action_log_prob`` inverts the squash (``atanh``) and applies
    the change-of-variables correction per dimension."""

    def __init__(self, features_dim: int, action_space: spaces.Box, hidden_dim: int = 32) -> None:
        super().__init__()
        low, high = _bounds(action_space)
        self.mean_net = nn.Sequential(
            nn.Linear(features_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, low.numel())
        )
        # Not called ``log_std``: that name opts into the Gaussian-only
        # features (adaptive_kl), which read a Normal over the action itself.
        self.pre_squash_log_std = nn.Parameter(torch.full((low.numel(),), -0.5))
        self.register_buffer("half_range", (high - low) / 2.0)
        self.register_buffer("middle", (high + low) / 2.0)

    def _pre_squash_distribution(self, features: torch.Tensor) -> Normal:
        return Normal(self.mean_net(features), self.pre_squash_log_std.exp())

    def _squash(self, pre_squash: torch.Tensor) -> torch.Tensor:
        return self.middle + self.half_range * torch.tanh(pre_squash)

    def _log_prob(self, distribution: Normal, pre_squash: torch.Tensor, sum_dims: bool) -> torch.Tensor:
        jacobian = self.half_range * (1.0 - torch.tanh(pre_squash) ** 2)
        per_dim = distribution.log_prob(pre_squash) - torch.log(jacobian + _ATANH_EPS)
        return per_dim.sum(dim=-1, keepdim=True) if sum_dims else per_dim

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        distribution = self._pre_squash_distribution(features)
        pre_squash = distribution.mean if deterministic else distribution.sample()
        log_prob = self._log_prob(distribution, pre_squash, sum_dims)
        return self._squash(pre_squash), log_prob, None

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        unit = ((actions - self.middle) / self.half_range).clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
        distribution = self._pre_squash_distribution(features)
        return self._log_prob(distribution, torch.atanh(unit), sum_dims), None

    def deterministic_action(self, features):
        return self._squash(self.mean_net(features))

    def clamp_action(self, actions):
        # Clip in the squash's unit coordinates, the head's own action frame.
        unit = ((actions - self.middle) / self.half_range).clamp(-1.0, 1.0)
        return self.middle + self.half_range * unit


def make_tanh_gaussian_ppo_actor(features_dim: int, action_space: spaces.Box) -> nn.Module:
    """``--actor-factory`` for PPO."""
    return TanhGaussianPPOActor(features_dim, action_space)


class SquashedSACActor(SACActorBase):
    """A ``SACActor`` with a mean/log-std MLP head and a tanh squash."""

    def __init__(self, features_dim: int, action_space: spaces.Box, hidden_dim: int = 32) -> None:
        super().__init__()
        low, high = _bounds(action_space)
        self.head = nn.Sequential(
            nn.Linear(features_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * low.numel())
        )
        self.register_buffer("half_range", (high - low) / 2.0)
        self.register_buffer("middle", (high + low) / 2.0)

    def _mean_log_std(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.head(features).chunk(2, dim=-1)
        return mean, log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)

    def action_log_prob(self, features):
        mean, log_std = self._mean_log_std(features)
        distribution = Normal(mean, log_std.exp())
        pre_squash = distribution.rsample()
        squashed = torch.tanh(pre_squash)
        log_prob = distribution.log_prob(pre_squash) - torch.log(
            self.half_range * (1.0 - squashed**2) + _ATANH_EPS
        )
        return self.middle + self.half_range * squashed, log_prob.sum(dim=-1, keepdim=True)

    def deterministic_action(self, features):
        mean, _ = self._mean_log_std(features)
        return self.middle + self.half_range * torch.tanh(mean)


def make_squashed_sac_actor(features_dim: int, action_space: spaces.Box) -> nn.Module:
    """``--actor-factory`` for the SAC family (SAC, RLPD, CQL, Cal-QL, WSRL, ...)."""
    return SquashedSACActor(features_dim, action_space)


class StateBackboneExtractor(BaseFeaturesExtractor):
    """A ``--actor-extractor``: concatenates the state keys of the
    observation and maps them through a small MLP to ``features_dim``
    features. A real package would load a pretrained backbone here."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 24) -> None:
        super().__init__(observation_space, features_dim)
        self.keys = tuple(observation_space.spaces)
        input_dim = sum(int(np.prod(observation_space.spaces[key].shape)) for key in self.keys)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, features_dim), nn.LayerNorm(features_dim), nn.GELU()
        )
        self.input_scale = 1.0 / math.sqrt(input_dim)

    def forward(self, obs) -> torch.Tensor:
        flat = torch.cat([obs[key].flatten(start_dim=1).float() for key in self.keys], dim=-1)
        return self.backbone(flat * self.input_scale)
