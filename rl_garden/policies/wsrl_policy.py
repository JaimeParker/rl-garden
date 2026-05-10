"""WSRL actor/critic policy with Q-ensemble (REDQ) and CQL support.

Extends SACPolicy-style interfaces to support:
  - Variable n_critics (2-10+) for REDQ ensemble
  - Critic subsampling for efficient target computation
  - Optional CQL alpha Lagrange multiplier for auto-tuning
  - Batch Q-value evaluation for CQL loss computation
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
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
LOG_STD_MIN = -20.0


def _softplus_inverse(x: float) -> float:
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
    """SAC entropy coefficient via softplus (matches GeqLagrangeMultiplier)."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        if init_value <= 0:
            raise ValueError("Temperature init value must be positive.")
        self.log_alpha = nn.Parameter(
            torch.tensor(_softplus_inverse(init_value), dtype=torch.float32)
        )

    def forward(self) -> torch.Tensor:
        return F.softplus(self.log_alpha)


class WSRLPolicy(BasePolicy):
    """WSRL policy with Q-ensemble and optional CQL alpha Lagrange multiplier."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        features_extractor: BaseFeaturesExtractor,
        net_arch: Sequence[int] | dict[str, Sequence[int]] = (256, 256),
        n_critics: int = 10,
        critic_subsample_size: Optional[int] = 2,
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
        actor_hidden_dims: Optional[Sequence[int]] = None,
        critic_hidden_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        assert isinstance(action_space, spaces.Box), "WSRL requires a Box action space."
        assert n_critics >= 2, f"n_critics must be >= 2, got {n_critics}"
        if critic_subsample_size is not None:
            assert critic_subsample_size <= n_critics, (
                f"critic_subsample_size ({critic_subsample_size}) must be <= n_critics ({n_critics})"
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.n_critics = n_critics
        self.critic_subsample_size = critic_subsample_size

        if actor_hidden_dims is not None or critic_hidden_dims is not None:
            actor_arch = list(actor_hidden_dims if actor_hidden_dims is not None else ())
            critic_arch = list(critic_hidden_dims if critic_hidden_dims is not None else actor_arch)
        else:
            actor_arch, critic_arch = get_actor_critic_arch(net_arch)

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
            log_std_mode="clamp",
            log_std_min=LOG_STD_MIN,
            log_std_max=LOG_STD_MAX,
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
        # Separate target critic
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

        # Optional CQL alpha Lagrange multiplier
        self.use_cql_alpha_lagrange = use_cql_alpha_lagrange
        if use_cql_alpha_lagrange:
            self.cql_alpha_lagrange = CQLAlphaLagrange(init_value=cql_alpha_lagrange_init)
        else:
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

    # --- helpers for WSRL.train() ---

    def actor_action_log_prob(
        self,
        obs: Obs,
        stop_gradient: bool = False,
        detach_encoder: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy and return action, log_prob, features."""
        if detach_encoder is not None:
            stop_gradient = detach_encoder
        features = self.extract_features(obs, stop_gradient=stop_gradient)
        action, log_prob = self.actor.action_log_prob(features)
        return action, log_prob, features

    def q_values(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Returns tuple of Q-values from all critics."""
        net = self.critic_target if target else self.critic
        return net(features, actions)

    def q_values_all(
        self, features: torch.Tensor, actions: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        """Returns stacked Q-values as tensor of shape (n_critics, batch_size, 1)."""
        net = self.critic_target if target else self.critic
        return net.forward_all(features, actions)

    def q_values_subsampled(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        """Returns subsampled Q-values for REDQ target computation.

        Args:
            features: State features
            actions: Actions to evaluate
            subsample_size: Number of critics to subsample. ``None`` uses all
                critics; callers that want REDQ subsampling should pass
                ``self.critic_subsample_size`` explicitly.
            target: Use target network (default: True)

        Returns:
            Tensor of shape (critic_count, batch_size, 1)
        """
        # Get all Q-values
        q_all = self.q_values_all(features, actions, target=target)  # (n_critics, batch, 1)

        # Subsample critics
        if subsample_size is not None and subsample_size < self.n_critics:
            indices = torch.randint(
                0, self.n_critics, (subsample_size,), device=q_all.device
            )
            q_subsampled = q_all[indices]  # (subsample_size, batch, 1)
        else:
            q_subsampled = q_all

        return q_subsampled

    def min_q_value(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        subsample_size: Optional[int] = None,
        target: bool = True,
    ) -> torch.Tensor:
        """Returns minimum Q-value across (subsampled) ensemble.

        Args:
            features: State features
            actions: Actions to evaluate
            subsample_size: Number of critics to subsample. ``None`` uses all critics.
            target: Use target network (default: True)

        Returns:
            Tensor of shape (batch_size, 1)
        """
        q_subsampled = self.q_values_subsampled(
            features, actions, subsample_size=subsample_size, target=target
        )
        return q_subsampled.min(dim=0)[0]  # (batch, 1)

    # --- CQL alpha Lagrange multiplier ---

    def get_cql_alpha(self) -> torch.Tensor:
        """Returns CQL alpha value (either from Lagrange multiplier or fixed)."""
        if self.use_cql_alpha_lagrange:
            return self.cql_alpha_lagrange()
        else:
            raise ValueError("CQL alpha Lagrange multiplier not enabled")

    # --- parameter groups for optimizers ---

    def critic_and_encoder_parameters(self):
        """Encoder trained via Q-loss (matches sac_rgbd.py pattern)."""
        yield from self.critic.parameters()
        yield from self.features_extractor.parameters()

    def actor_parameters(self):
        """Actor-only; RGBD actor path uses stop_gradient on image encodings."""
        yield from self.actor.parameters()

    def cql_alpha_lagrange_parameters(self):
        """CQL alpha Lagrange multiplier parameters."""
        if self.use_cql_alpha_lagrange:
            yield from self.cql_alpha_lagrange.parameters()
        else:
            return iter([])
