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
from rl_garden.networks.actor_critic import (
    CriticImpl,
    SquashedGaussianActor as Actor,
    get_actor_critic_arch,
)
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
        critic_impl: CriticImpl = "vmap",
        actor_feature_dim: Optional[int] = None,
        critic_spatial_emb_dim: int = 1024,
        critic_use_layer_norm: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            critic_impl=critic_impl,
            actor_feature_dim=actor_feature_dim,
            critic_spatial_emb_dim=critic_spatial_emb_dim,
            critic_use_layer_norm=critic_use_layer_norm,
        )
        # Rebuild actor: base_actions are appended after the adapter (if any).
        actor_arch, _ = get_actor_critic_arch(net_arch)
        action_dim = int(np.prod(action_space.shape))
        self.actor = Actor(
            self._actor_fd + action_dim,  # post-adapter dim + base_action_dim
            action_space,
            hidden_dims=actor_arch,
            log_std_mode="tanh",
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
        )

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
        adapted = self._transform_features_for_actor(features)
        actor_input = torch.cat([adapted, base_actions], dim=-1)
        if deterministic:
            return self.actor.deterministic_action(actor_input)
        action, _ = self.actor.action_log_prob(actor_input)
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
        adapted = self._transform_features_for_actor(features)
        actor_input = torch.cat([adapted, base_actions], dim=-1)
        unit_residual_action, log_prob = self.actor.action_log_prob(actor_input)
        return unit_residual_action, log_prob, features

    def actor_diagnostics(
        self, obs, base_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            features = self.extract_features(obs, stop_gradient=True)
            adapted = self._transform_features_for_actor(features)
            actor_input = torch.cat([adapted, base_actions], dim=-1)
            mean, log_std = self.actor(actor_input)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            gaussian_term = normal.log_prob(x_t).sum(-1)
            tanh_correction = torch.log(
                self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6
            ).sum(-1)
            return {
                "policy_std": std.mean(),
                "action_saturation": torch.tanh(mean).abs().mean(),
                "entropy_gaussian": -gaussian_term.mean(),
                "tanh_correction": tanh_correction.mean(),
            }
