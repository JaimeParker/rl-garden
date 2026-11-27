"""MeanFlowBC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import Activation, KernelInit


@dataclass
class MeanFlowBCParams(OfflineOptimizationArgs):
    """Constructor defaults for MeanFlowBC."""

    actor_lr: float = 0.0003
    net_arch: Optional[tuple[int, ...]] = None
    num_sample_steps: int = 1
    mode: Literal["meanflow", "i-meanflow"] = 'i-meanflow'
    time_dist_mu: float = 0.4
    time_dist_sigma: float = 1.0
    adaptive_l2_gamma: float = 0.0
    adaptive_l2_c: float = 0.01
    actor_use_layer_norm: bool = False
    kernel_init: Optional[KernelInit] = None
    activation_fn: Optional[Activation] = None
