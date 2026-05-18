"""IQL policy with a shared feature extractor.

The extractor is shared by actor, critic, and value networks. Critic/value
updates own the encoder; actor updates use detached features, matching the
existing RGBD SAC convention in rl-garden.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    BackboneType,
    EnsembleQCritic,
    KernelInit,
    SquashedGaussianActor,
    ValueNetwork,
)
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.sac_policy import LOG_STD_MAX, WSRL_LOG_STD_MIN


def get_iql_arch(
    net_arch: Sequence[int] | dict[str, Sequence[int]],
) -> tuple[list[int], list[int], list[int]]:
    """Resolve actor/critic/value hidden dims from an SB3-style spec."""
    if isinstance(net_arch, dict):
        if "pi" not in net_arch or "qf" not in net_arch:
            raise ValueError("net_arch dict must contain both 'pi' and 'qf' keys.")
        vf = net_arch.get("vf", net_arch["qf"])
        return list(net_arch["pi"]), list(net_arch["qf"]), list(vf)
    shared = list(net_arch)
    return shared, list(shared), list(shared)


class IQLPolicy(BasePolicy):
    """Actor, critic ensemble, and value head for Implicit Q-Learning."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        actor_use_layer_norm: bool = False,
        critic_use_layer_norm: bool = False,
        value_use_layer_norm: bool = False,
        actor_use_group_norm: bool = False,
        critic_use_group_norm: bool = False,
        value_use_group_norm: bool = False,
        num_groups: int = 32,
        actor_dropout_rate: Optional[float] = None,
        critic_dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        kernel_init: Optional[KernelInit] = None,
        backbone_type: BackboneType = "mlp",
        std_parameterization: Literal["exp", "uniform"] = "exp",
        log_std_mode: Literal["clamp", "tanh"] = "tanh",
        log_std_min: float = WSRL_LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "IQL requires a Box action space."
        if n_critics < 2:
            raise ValueError(f"n_critics must be >= 2, got {n_critics}.")
        if critic_subsample_size is not None and critic_subsample_size > n_critics:
            raise ValueError(
                f"critic_subsample_size ({critic_subsample_size}) must be <= "
                f"n_critics ({n_critics})."
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        actor_arch, critic_arch, value_arch = get_iql_arch(net_arch)
        fd = features_extractor.features_dim
        self.actor = SquashedGaussianActor(
            fd,
            action_space,
            hidden_dims=actor_arch,
            use_layer_norm=actor_use_layer_norm,
            use_group_norm=actor_use_group_norm,
            num_groups=num_groups,
            dropout_rate=actor_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            log_std_mode=log_std_mode,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.critic = EnsembleQCritic(
            fd,
            action_space,
            hidden_dims=critic_arch,
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
            hidden_dims=critic_arch,
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

        self.value = ValueNetwork(
            fd,
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

    def forward(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        return self.predict(obs, deterministic=deterministic)

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

    def q_values_subsampled(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        q_all = self.q_values_all(features, actions, target=target)
        if subsample_size is not None and subsample_size < self.n_critics:
            indices = torch.randint(
                0, self.n_critics, (subsample_size,), device=q_all.device
            )
            return q_all[indices]
        return q_all

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        return (
            self.q_values_subsampled(
                features, actions, subsample_size=subsample_size, target=target
            )
            .min(dim=0)
            .values
        )

    def critic_value_and_encoder_parameters(self):
        yield from self.critic.parameters()
        yield from self.value.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()
