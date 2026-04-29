"""SAC actor/critic policy with a shared features extractor.

Mirrors the architecture used by ManiSkill's sac.py (state) and sac_rgbd.py
(RGBD), reorganized SB3-style: the ``SACPolicy`` owns:
  - ``features_extractor`` (shared encoder; can be ``FlattenExtractor`` for
    state obs, ``CombinedExtractor`` for RGBD)
  - ``actor`` (MLP -> mean/log_std heads, tanh-squashed Normal)
  - ``critic`` (ensemble of Q-nets that *reuse* the same extractor)
  - ``critic_target`` (same, no grad)

Key RGBD detail, preserved from hil-serl/ManiSkill visual SAC:
  - Critic optimizer owns the encoder params (encoder learns via Q-loss).
  - Actor update extracts visual features with ``stop_gradient=True`` so the
    image encoder sees no gradients from the policy loss.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import EnsembleQCritic, SquashedGaussianActor, get_actor_critic_arch
from rl_garden.policies.base import BasePolicy

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


Actor = SquashedGaussianActor
ContinuousCritic = EnsembleQCritic


class SACPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 2,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "SAC requires a Box action space."
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor

        if actor_hidden_dims is not None or critic_hidden_dims is not None:
            # Backward-compatible path for direct policy construction.
            actor_arch = list(actor_hidden_dims if actor_hidden_dims is not None else ())
            critic_arch = list(critic_hidden_dims if critic_hidden_dims is not None else actor_arch)
        else:
            actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        fd = features_extractor.features_dim
        self.actor = Actor(
            fd,
            action_space,
            hidden_dims=actor_arch,
            std_parameterization="exp",
            log_std_mode="tanh",
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
        )
        self.critic = ContinuousCritic(
            fd,
            action_space,
            hidden_dims=critic_arch,
            n_critics=n_critics,
        )
        # Separate target critic, initialized to match critic.
        self.critic_target = ContinuousCritic(
            fd,
            action_space,
            hidden_dims=critic_arch,
            n_critics=n_critics,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    # --- feature extraction helpers ---

    def extract_features(
        self,
        obs: Obs,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

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
        self,
        obs: Obs,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_encoder is not None:
            stop_gradient = detach_encoder
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        action, log_prob = self.actor.action_log_prob(features)
        return action, log_prob, features

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, ...]:
        net = self.critic_target if target else self.critic
        return net(features, actions)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        # Encoder trained via Q-loss (matches sac_rgbd.py L581-L585).
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        # Actor-only; RGBD actor path uses stop_gradient on image encodings.
        yield from self.actor.parameters()
