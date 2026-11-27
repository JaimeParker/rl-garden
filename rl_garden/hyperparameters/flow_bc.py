"""FlowBC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import Activation, KernelInit


@dataclass
class FlowBCParams(OfflineOptimizationArgs):
    """Constructor defaults for FlowBC."""

    actor_lr: float = 0.0003
    net_arch: Optional[tuple[int, ...]] = None
    flow_steps: int = 10
    actor_use_layer_norm: bool = False
    kernel_init: Optional[KernelInit] = None
    activation_fn: Optional[Activation] = None
