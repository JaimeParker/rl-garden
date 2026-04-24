"""SAC actor/critic policy with a shared features extractor.

Mirrors the architecture used by ManiSkill's sac.py (state) and sac_rgbd.py
(RGBD), reorganized SB3-style: the ``SACPolicy`` owns:
  - ``features_extractor`` (shared encoder; can be ``FlattenExtractor`` for
    state obs, ``CombinedExtractor`` for RGBD)
  - ``actor`` (MLP -> mean/log_std heads, tanh-squashed Normal)
  - ``critic`` (2-ensemble of Q-nets that *reuse* the same extractor)
  - ``critic_target`` (same, no grad)

Key RGBD detail, preserved from sac_rgbd.py L696-L723:
  - Critic optimizer owns the encoder params (encoder learns via Q-loss).
  - Actor update passes ``detach_encoder=True`` so the encoder sees no
    gradients from the policy loss.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.policies.base import BasePolicy

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, last_act: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_dim
    for h in hidden:
        layers += [nn.Linear(c, h), nn.ReLU()]
        c = h
    layers.append(nn.Linear(c, out_dim))
    if last_act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        layers: list[nn.Module] = []
        c = features_dim
        for h in hidden_dims:
            layers += [nn.Linear(c, h), nn.ReLU()]
            c = h
        self.trunk = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(c, act_dim)
        self.fc_logstd = nn.Linear(c, act_dim)

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(features)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def action_log_prob(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self(features)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class ContinuousCritic(nn.Module):
    """Ensemble of Q(s, a) MLPs sharing a features extractor with the actor."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Sequence[int] = (256, 256, 256),
        n_critics: int = 2,
    ) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.q_nets = nn.ModuleList(
            [
                _mlp(features_dim + act_dim, hidden_dims, out_dim=1, last_act=False)
                for _ in range(n_critics)
            ]
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat([features, actions], dim=-1)
        return tuple(q(x) for q in self.q_nets)


class SACPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        actor_hidden_dims: Sequence[int] = (256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        n_critics: int = 2,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "SAC requires a Box action space."
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        fd = features_extractor.features_dim
        self.actor = Actor(fd, action_space, hidden_dims=actor_hidden_dims)
        self.critic = ContinuousCritic(
            fd, action_space, hidden_dims=critic_hidden_dims, n_critics=n_critics
        )
        # Separate target critic, initialized to match critic.
        self.critic_target = ContinuousCritic(
            fd, action_space, hidden_dims=critic_hidden_dims, n_critics=n_critics
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    # --- feature extraction helpers ---

    def extract_features(
        self, obs: Obs, detach: bool = False
    ) -> torch.Tensor:
        features = self.features_extractor(obs)
        if detach:
            features = features.detach()
        return features

    # --- public inference API ---

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        action, _ = self.actor.action_log_prob(features)
        return action

    # --- helpers for SAC.train() ---

    def actor_action_log_prob(
        self, obs: Obs, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, detach=detach_encoder)
        action, log_prob = self.actor.action_log_prob(features)
        return action, log_prob, features

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, ...]:
        net = self.critic_target if target else self.critic
        return net(features, actions)

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        # Encoder trained via Q-loss (matches sac_rgbd.py L581-L585).
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        # Actor-only; encoder is detached on the actor path.
        yield from self.actor.parameters()
