"""TD3BC learning hyperparameters, shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.shared import (
    OfflineDeterministicActorCriticArgs,
    OfflineOptimizationArgs,
)
from rl_garden.networks.mlp import KernelInit


@dataclass
class TD3CoreParams(OfflineDeterministicActorCriticArgs, OfflineOptimizationArgs):
    """Learning settings consumed by TD3-BC, ReBRAC and SPOT."""

    # Target critic Polyak update coefficient.
    tau: float = 0.005

    # Actor learning rate.
    actor_lr: float = 3e-4

    # Critic learning rate.
    critic_lr: float = 3e-4

    # Target action smoothing noise standard deviation.
    policy_noise: float = 0.2

    # Target and exploration noise clipping bound.
    noise_clip: float = 0.5

    # Actor and target update interval.
    policy_freq: int = 2

    # Hidden layer widths; None selects two layers of width 256.
    net_arch: Optional[tuple[int, ...]] = None

    # Network weight initialization.
    kernel_init: Optional[KernelInit] = None


@dataclass
class TD3BCParams(TD3CoreParams):
    """TD3-BC's behavior-cloning objective weight."""

    # Normalized behavior-cloning objective weight.
    alpha: float = 2.5
