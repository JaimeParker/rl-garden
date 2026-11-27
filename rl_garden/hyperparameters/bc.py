"""BC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import KernelInit


@dataclass
class BCParams(OfflineOptimizationArgs):
    """Constructor defaults for BC."""

    actor_lr: float = 0.0003
    net_arch: Optional[tuple[int, ...]] = None
    actor_use_layer_norm: bool = False
    actor_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: Literal['mlp', 'mlp_resnet'] = 'mlp'
    std_parameterization: Literal['exp', 'uniform'] = 'exp'
    tanh_squash: bool = True
