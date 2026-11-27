"""On-policy learning hyperparameters, shared by algorithms and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import KernelInit


@dataclass
class PolicyDistillationParams:
    """Constructor defaults for PolicyDistillation."""

    num_learning_epochs: int = 5
    num_minibatches: int = 4
    loss_type: Literal['mse', 'huber'] = 'mse'
    actor_lr: float = 0.0003
    weight_decay: float = 0.0
    use_adamw: bool = False
    max_grad_norm: Optional[float] = None
    net_arch: Optional[tuple[int, ...]] = None
    actor_use_layer_norm: bool = False
    actor_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: Literal['mlp', 'mlp_resnet'] = 'mlp'
    std_parameterization: Literal['exp', 'uniform'] = 'exp'
    tanh_squash: bool = True
