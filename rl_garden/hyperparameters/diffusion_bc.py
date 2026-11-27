"""DiffusionBC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import Activation, KernelInit


@dataclass
class DiffusionPolicyParams:
    """Diffusion and network settings shared by teacher and student policies."""

    horizon_steps: int = 4
    cond_steps: int = 1
    denoising_steps: int = 20
    mlp_dims: Optional[tuple[int, ...]] = None
    activation_fn: Optional[Activation] = 'relu'
    residual_style: bool = True
    time_dim: int = 16
    kernel_init: Optional[KernelInit] = None
    denoised_clip_value: Optional[float] = 1.0
    randn_clip_value: float = 10.0
    final_action_clip_value: Optional[float] = None
    min_sampling_denoising_std: float = 0.1


@dataclass
class DiffusionBCParams(DiffusionPolicyParams):
    """Constructor defaults for DiffusionBC."""

    actor_lr: float = 0.001
    weight_decay: float = 1e-06
    lr_schedule: Literal['constant', 'linear_warmup', 'warmup_cosine'] = 'constant'
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
    ema_decay: float = 0.995
    ema_update_every: int = 10
    ema_start_step: Optional[int] = None
