"""AWAC policy: normalized obs + unsquashed Gaussian actor + twin-Q critic.

Box observations only (matches CORL's D4RL MuJoCo scope). No actor target --
AWAC's critic backup samples ``next_action`` from the current actor, not a
target actor (see ``rl_garden.algorithms.awac`` for why).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.obs_normalization import ObsNormalizingMixin
from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import EnsembleQCritic, KernelInit, UnsquashedGaussianActor
from rl_garden.networks.actor_critic import BackboneType
from rl_garden.policies.base import BasePolicy


class AWACPolicy(ObsNormalizingMixin, BasePolicy):
    """Unsquashed Gaussian actor (no target) + twin-Q critic (with target)."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] = (256, 256, 256),
        n_critics: int = 2,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, spaces.Box), (
            "AWACPolicy requires a Box observation space."
        )
        assert isinstance(action_space, spaces.Box), "AWAC requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self._register_obs_normalizer(int(observation_space.shape[0]))

        fd = features_extractor.features_dim
        net_arch = list(net_arch)

        self.actor = UnsquashedGaussianActor(
            fd,
            action_space,
            hidden_dims=net_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
        )

        self.critic = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=net_arch,
            n_critics=n_critics,
            use_layer_norm=critic_use_layer_norm,
            use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        obs = self._normalize_obs(obs)
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.deterministic_action(features)
        action, _ = self.actor.action_log_prob(features)
        return action

    def behavior_log_prob(
        self, obs: Obs, actions: torch.Tensor, stop_gradient: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        return self.actor.evaluate_action_log_prob(
            features, actions
        ), self.actor.deterministic_action(features)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def min_q_value(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = True
    ) -> torch.Tensor:
        return self.q_values_all(features, actions, target=target).min(dim=0).values

    def actor_parameters(self):
        yield from self.actor.parameters()

    def critic_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()
