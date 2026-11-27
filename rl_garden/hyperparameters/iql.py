"""IQL learning hyperparameters shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import (
    OfflineDeterministicActorCriticArgs,
    OfflineOptimizationArgs,
)
from rl_garden.networks.mlp import KernelInit


@dataclass
class IQLParams(OfflineDeterministicActorCriticArgs, OfflineOptimizationArgs):
    """Defaults of the shared expectile-value and advantage-weighted actor core."""

    # Network weight initialization.
    kernel_init: Optional[KernelInit] = None

    # Target critic Polyak update coefficient.
    tau: float = 0.005

    # Actor learning rate.
    actor_lr: float = 3e-4

    # Joint critic and value learning rate.
    critic_value_lr: float = 3e-4

    # Actor schedule; None uses the shared optimizer schedule.
    actor_lr_schedule: Optional[Literal["constant", "linear_warmup", "warmup_cosine"]] = None

    # Actor warmup updates; None uses the shared warmup count.
    actor_lr_warmup_steps: Optional[int] = None

    # Actor decay updates; None uses the shared decay count.
    actor_lr_decay_steps: Optional[int] = None

    # Minimum actor learning-rate fraction; None uses the shared fraction.
    actor_lr_min_ratio: Optional[float] = None

    # Expectile of the value regression.
    expectile: float = 0.7

    # Multiplier in the exponentiated advantage weights.
    temperature: float = 3.0

    # Maximum exponentiated advantage weight.
    adv_clip_max: float = 100.0

    # Squashed Gaussian or bounded unsquashed Gaussian actor.
    actor_distribution: Literal["squashed", "unsquashed"] = "squashed"

    # Shared widths or pi/qf/vf widths; None selects two layers of width 256.
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = None

    # Number of target critics sampled; None uses the full ensemble.
    critic_subsample_size: Optional[int] = None

    # Apply layer normalization in the value network.
    value_use_layer_norm: bool = False

    # Apply group normalization in the value network.
    value_use_group_norm: bool = False

    # Value-network dropout probability, or None to disable.
    value_dropout_rate: Optional[float] = None

    # Gaussian standard-deviation parameterization.
    std_parameterization: Literal["exp", "uniform"] = "exp"
