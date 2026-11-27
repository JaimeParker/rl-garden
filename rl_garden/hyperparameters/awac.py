"""AWAC learning hyperparameters, shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import (
    OfflineDeterministicActorCriticArgs,
    OfflineOptimizationArgs,
)
from rl_garden.networks.mlp import KernelInit


@dataclass
class AWACParams(OfflineDeterministicActorCriticArgs, OfflineOptimizationArgs):
    """AWAC defaults matching the learning core."""

    # Target critic Polyak update coefficient.
    tau: float = 5e-3
    # Actor learning rate.
    actor_lr: float = 3e-4
    # Critic learning rate.
    critic_lr: float = 3e-4
    # Advantage-weighting temperature.
    awac_lambda: float = 1.0
    # Maximum exponentiated advantage weight.
    exp_adv_max: float = 100.0
    # Hidden layer widths; None selects three layers of width 256.
    net_arch: Optional[tuple[int, ...]] = None
    # Gaussian standard-deviation parameterization.
    std_parameterization: Literal["exp", "uniform"] = "exp"
    # AWAC accepts all constructor initializers, including near-zero output initialization.
    kernel_init: Optional[KernelInit] = None
