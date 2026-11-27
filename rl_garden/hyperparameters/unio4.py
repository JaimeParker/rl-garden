"""UniO4 learning parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks.mlp import KernelInit
from rl_garden.networks.actor_critic import BackboneType

from rl_garden.hyperparameters.bppo import BPPOCriticParams


@dataclass
class UniO4Params(BPPOCriticParams):
    """Actor ensemble and PPO improvement settings."""

    num_policies: int = 4
    bc_ensemble_steps: int = 400000
    alpha_bc: float = 0.1
    actor_lr: float = 0.0001
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    clip_ratio: float = 0.25
    clip_decay: float = 0.96
    clip_decay_steps: int = 200
    entropy_weight: float = 0.0
    omega: float = 0.7
    grad_clip_norm: Optional[float] = None
    use_layer_norm: bool = False
    use_group_norm: bool = False
    num_groups: int = 32
    dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: BackboneType = 'mlp'
    std_parameterization: Literal['exp', 'uniform'] = 'exp'
