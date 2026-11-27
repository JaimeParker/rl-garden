"""On-policy learning hyperparameters, shared by algorithms and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import KernelInit, Activation


@dataclass
class ChainPPOCoreParams:
    """Constructor defaults for ChainPPOCore."""

    actor_mlp_dims: Optional[tuple[int, ...]] = None
    actor_activation_fn: Optional[Activation] = None
    critic_mlp_dims: Optional[tuple[int, ...]] = None
    critic_activation_fn: Optional[Activation] = None
    kernel_init: Optional[KernelInit] = None
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    weight_decay: float = 0.0
    lr_schedule: Literal['constant', 'linear_warmup', 'warmup_cosine'] = 'constant'
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
    critic_warmup_updates: int = 0
    update_epochs: int = 5
    update_batch_size: int = 50000
    norm_adv: bool = True
    clip_vloss_coef: Optional[float] = None
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    vf_coef: float = 0.5
    target_kl: Optional[float] = 1.0

@dataclass
class DPPOParams(ChainPPOCoreParams):
    """Constructor defaults for DPPO."""

    horizon_steps: int = 4
    act_steps: int = 4
    denoising_steps: int = 20
    ft_denoising_steps: int = 10
    actor_activation_fn: Optional[Activation] = 'relu'
    actor_residual_style: bool = True
    critic_activation_fn: Optional[Activation] = 'mish'
    critic_residual_style: bool = True
    time_dim: int = 16
    denoised_clip_value: Optional[float] = 1.0
    randn_clip_value: float = 3.0
    final_action_clip_value: Optional[float] = None
    min_sampling_denoising_std: float = 0.1
    min_logprob_denoising_std: float = 0.1
    actor_lr: float = 0.0001
    gamma_denoising: float = 0.99
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 0.01
    clip_ploss_coef_rate: float = 3.0
    reward_horizon: Optional[int] = None

@dataclass
class FlowPPOParams(ChainPPOCoreParams):
    """Constructor defaults for FlowPPO."""

    horizon_length: int = 1
    flow_steps: int = 10
    sde_type: Literal['sde', 'cps'] = 'cps'
    noise_level: float = 0.7
    clip_std_min: float = 0.0067
    sigma_safe_max: float = 0.9
    logprob_mode: Literal['gaussian', 'pseudo'] = 'gaussian'
    clip_coef: float = 0.2

@dataclass
class DiffusionCMDistillOnlineParams(DPPOParams):
    """Constructor defaults for DiffusionCMDistillOnline."""

    cm_mlp_dims: Optional[tuple[int, ...]] = None
    cm_lr: float = 0.0001
    cm_ema_decay: float = 0.95
    cm_grad_clip_norm: Optional[float] = 1.0
    cm_sigma_data: float = 0.5
    cm_timestep_scaling: float = 0.1
