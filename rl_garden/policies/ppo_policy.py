"""Actor-critic policy for PPO with pluggable feature extractors."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    BackboneType,
    DiagGaussianActor,
    KernelInit,
    ValueNetwork,
)
from rl_garden.policies.base import BasePolicy


def get_ppo_arch(
    net_arch: Sequence[int] | dict[str, Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Resolve PPO actor/value hidden dims from an SB3-style net_arch spec."""
    if isinstance(net_arch, dict):
        if "pi" not in net_arch or "vf" not in net_arch:
            raise ValueError("PPO net_arch dict must contain both 'pi' and 'vf' keys.")
        return list(net_arch["pi"]), list(net_arch["vf"])
    shared = list(net_arch)
    return shared, list(shared)


class PPOPolicy(BasePolicy):
    """Continuous-action actor-critic policy used by PPO."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        *,
        features_dim: Optional[int] = None,
        log_std_init: float = -0.5,
        actor_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        value_features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> None:
        super().__init__()
        if not isinstance(action_space, spaces.Box):
            raise TypeError("PPOPolicy only supports Box action spaces.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.value_features_extractor = value_features_extractor
        actor_arch, value_arch = get_ppo_arch(net_arch)
        fd = features_dim if features_dim is not None else features_extractor.features_dim
        value_fd = (
            value_features_extractor.features_dim
            if value_features_extractor is not None
            else fd
        )
        self.actor = DiagGaussianActor(
            fd,
            action_space,
            hidden_dims=actor_arch,
            log_std_init=log_std_init,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )
        self.value_net = ValueNetwork(
            value_fd,
            value_arch,
            use_layer_norm=value_use_layer_norm,
            use_group_norm=value_use_group_norm,
            num_groups=num_groups,
            dropout_rate=value_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
        )

    def extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._extract_features(obs, stop_gradient=stop_gradient)

    def _value_encoder(self) -> BaseFeaturesExtractor:
        return (
            self.value_features_extractor
            if self.value_features_extractor is not None
            else self.features_extractor
        )

    def value_extract_features(self, obs: Obs, stop_gradient: bool = False) -> torch.Tensor:
        return self._value_encoder().extract(obs, stop_gradient=stop_gradient)

    def forward(
        self,
        obs: Obs,
        deterministic: bool = False,
        *,
        stop_gradient_actor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = self.extract_features(obs, stop_gradient=stop_gradient_actor)
        value_features = self.value_extract_features(obs, stop_gradient=False)
        actions, log_prob, entropy = self.actor.action_log_prob(
            actor_features, deterministic=deterministic
        )
        values = self.value_net(value_features)
        return actions, values, log_prob, entropy

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs)
        if deterministic:
            return self.actor.clamp_action(self.actor.deterministic_action(features))
        action, _, _ = self.actor.action_log_prob(features, deterministic=False)
        return self.actor.clamp_action(action)

    def predict_values(self, obs: Obs) -> torch.Tensor:
        return self.value_net(self.value_extract_features(obs, stop_gradient=False))

    def evaluate_actions(
        self,
        obs: Obs,
        actions: torch.Tensor,
        *,
        stop_gradient_actor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features = self.extract_features(obs, stop_gradient=stop_gradient_actor)
        value_features = self.value_extract_features(obs, stop_gradient=False)
        log_prob, entropy = self.actor.evaluate_action_log_prob(actor_features, actions)
        values = self.value_net(value_features)
        return values, log_prob, entropy

    def clamp_action(self, actions: torch.Tensor) -> torch.Tensor:
        return self.actor.clamp_action(actions)
