"""Compatibility shim for the WSRL policy.

The generic SAC-family policy now owns REDQ critic ensembles, critic
subsampling, modern MLP options, and optional CQL alpha Lagrange state.
``WSRLPolicy`` remains as a thin subclass so existing imports and checkpoint
module paths continue to work while WSRL is migrated onto the SAC-family core.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import BackboneType, KernelInit
from rl_garden.policies.sac_policy import (
    CQLAlphaLagrange,
    SACPolicy,
    TemperatureLagrange,
    WSRL_LOG_STD_MIN,
)


class WSRLPolicy(SACPolicy):
    """WSRL-compatible SACPolicy defaults."""

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
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            net_arch=net_arch,
            n_critics=n_critics,
            critic_subsample_size=critic_subsample_size,
            use_cql_alpha_lagrange=use_cql_alpha_lagrange,
            cql_alpha_lagrange_init=cql_alpha_lagrange_init,
            actor_use_layer_norm=actor_use_layer_norm,
            critic_use_layer_norm=critic_use_layer_norm,
            actor_use_group_norm=actor_use_group_norm,
            critic_use_group_norm=critic_use_group_norm,
            num_groups=num_groups,
            actor_dropout_rate=actor_dropout_rate,
            critic_dropout_rate=critic_dropout_rate,
            kernel_init=kernel_init,
            backbone_type=backbone_type,
            std_parameterization=std_parameterization,
            log_std_mode="clamp",
            log_std_min=WSRL_LOG_STD_MIN,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )


__all__ = [
    "CQLAlphaLagrange",
    "TemperatureLagrange",
    "WSRLPolicy",
]
