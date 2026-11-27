"""A2ABC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import Activation, KernelInit


@dataclass
class A2ABCParams:
    """Constructor defaults for A2ABC."""

    horizon_steps: int = 8
    cond_steps: int = 8
    latent_dim: int = 512
    cnn_num_layers: int = 3
    cnn_hidden_channels: int = 512
    cnn_kernel_size: int = 5
    cnn_activation_fn: Optional[Activation] = 'relu'
    decoder_net_arch: Optional[tuple[int, ...]] = None
    decoder_activation_fn: Optional[Activation] = None
    decoder_kernel_init: Optional[KernelInit] = None
    flow_hidden_dims: Optional[tuple[int, ...]] = None
    flow_use_layer_norm: bool = False
    flow_kernel_init: Optional[KernelInit] = None
    flow_activation_fn: Optional[Activation] = None
    num_sampling_steps: int = 6
    consistency_weight: float = 1.0
    enc_recon_weight: float = 0.5
    flow_recon_weight: float = 0.5
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.1
    actor_lr: float = 0.001
    weight_decay: float = 1e-06
    lr_schedule: Literal['constant', 'linear_warmup', 'warmup_cosine'] = 'constant'
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
