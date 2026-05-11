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

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.common.types import Obs
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import (
    BackboneType,
    EnsembleQCritic,
    KernelInit,
    SquashedGaussianActor,
    get_actor_critic_arch,
)
from rl_garden.policies.base import BasePolicy

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
WSRL_LOG_STD_MIN = -20.0


Actor = SquashedGaussianActor
ContinuousCritic = EnsembleQCritic


def _softplus_inverse(x: float) -> float:
    import numpy as np

    return float(np.log(np.expm1(x)))


class CQLAlphaLagrange(nn.Module):
    """Lagrange multiplier for auto-tuning CQL alpha."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("CQL alpha Lagrange init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class TemperatureLagrange(nn.Module):
    """SAC entropy coefficient via softplus."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("Temperature init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class SACPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256, 256),
        n_critics: int = 2,
        critic_subsample_size: Optional[int] = None,
        use_cql_alpha_lagrange: bool = False,
        cql_alpha_lagrange_init: float = 1.0,
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
        log_std_mode: Literal["clamp", "tanh"] = "tanh",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        del use_cql_alpha_lagrange, cql_alpha_lagrange_init
        super().__init__()
        assert isinstance(action_space, spaces.Box), "SAC requires a Box action space."
        assert n_critics >= 2, f"n_critics must be >= 2, got {n_critics}"
        if critic_subsample_size is not None:
            assert critic_subsample_size <= n_critics, (
                f"critic_subsample_size ({critic_subsample_size}) must be <= "
                f"n_critics ({n_critics})"
            )
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

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
        self.critic = ContinuousCritic(
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
        # Separate target critic, initialized to match critic.
        self.critic_target = ContinuousCritic(
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

        self.use_cql_alpha_lagrange = False
        self.cql_alpha_lagrange = None

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
        return self.q_values_subsampled(
            features, actions, subsample_size=subsample_size, target=target
        ).min(dim=0).values

    # --- CQL alpha Lagrange multiplier ---

    def get_cql_alpha(self) -> torch.Tensor:
        raise ValueError(
            "CQL alpha Lagrange multiplier is owned by CQL/CalQL algorithms, "
            "not SACPolicy."
        )

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        # Encoder trained via Q-loss (matches sac_rgbd.py L581-L585).
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        # Actor-only; RGBD actor path uses stop_gradient on image encodings.
        yield from self.actor.parameters()

    def cql_alpha_lagrange_parameters(self):
        if self.use_cql_alpha_lagrange:
            assert self.cql_alpha_lagrange is not None
            yield from self.cql_alpha_lagrange.parameters()
