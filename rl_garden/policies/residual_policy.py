"""Residual SAC policy.

The critic evaluates normalized final actions. The actor predicts a unit
residual action from features concatenated with the normalized base action.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import SquashedGaussianActor as Actor
from rl_garden.networks import get_actor_critic_arch
from rl_garden.policies.sac_policy import LOG_STD_MAX, LOG_STD_MIN, SACPolicy


class ResidualSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
        )
        actor_arch, _ = get_actor_critic_arch(net_arch)
        action_dim = int(np.prod(action_space.shape))
        self.actor = Actor(
            features_extractor.features_dim + action_dim,
            action_space,
            hidden_dims=actor_arch,
            log_std_mode="tanh",
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
        )

    @staticmethod
    def _actor_features(features: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([features, base_actions], dim=-1)

    def predict(
        self,
        obs,
        deterministic: bool = False,
        *,
        base_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_actions is None:
            raise ValueError("ResidualSACPolicy.predict requires base_actions.")
        features = self.extract_features(obs)
        actor_features = self._actor_features(features, base_actions)
        if deterministic:
            return self.actor.deterministic_action(actor_features)
        action, _ = self.actor.action_log_prob(actor_features)
        return action

    def actor_action_log_prob(
        self,
        obs,
        base_actions: torch.Tensor,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detach_encoder is not None:
            stop_gradient = detach_encoder
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        actor_features = self._actor_features(features, base_actions)
        unit_residual_action, log_prob = self.actor.action_log_prob(actor_features)
        return unit_residual_action, log_prob, features
