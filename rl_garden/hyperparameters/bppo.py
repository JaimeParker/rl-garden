"""BPPO learning parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks.mlp import KernelInit
from rl_garden.networks.actor_critic import BackboneType


@dataclass
class BPPOCriticParams:
    """The shared V/Q critic and its optimizers."""

    critic_warmup_steps: int = 2000000
    value_lr: float = 0.0001
    q_lr: float = 0.0001
    tau: float = 0.005
    target_update_freq: int = 2
    value_hidden_dims: tuple[int, ...] = (512, 512, 512)
    q_hidden_dims: tuple[int, ...] = (1024, 1024)
    weight_decay: float = 0.0
    use_adamw: bool = False


@dataclass
class BPPOParams(BPPOCriticParams):
    """Single-actor PPO improvement settings."""

    actor_lr: float = 0.0001
    actor_hidden_dims: tuple[int, ...] = (1024, 1024)
    clip_ratio: float = 0.25
    clip_decay: float = 0.96
    clip_decay_steps: int = 200
    entropy_weight: float = 0.0
    omega: float = 0.9
    grad_clip_norm: Optional[float] = None
    use_layer_norm: bool = False
    use_group_norm: bool = False
    num_groups: int = 32
    dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: BackboneType = 'mlp'
    std_parameterization: Literal['exp', 'uniform'] = 'exp'
