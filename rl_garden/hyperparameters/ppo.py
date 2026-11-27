"""On-policy learning hyperparameters, shared by algorithms and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import KernelInit


@dataclass
class PPOParams:
    """Constructor defaults for PPO."""

    learning_rate: float = 0.0003
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    anneal_lr: bool = False
    weight_decay: float = 0.0
    use_adamw: bool = False
    lr_schedule: Literal['constant', 'linear_warmup', 'warmup_cosine', 'adaptive_kl'] = 'constant'
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    desired_kl: float = 0.01
    adaptive_lr_min: float = 1e-05
    adaptive_lr_max: float = 0.01
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = None
    log_std_init: float = -0.5
    log_std_init_per_dim: Optional[tuple[float, ...]] = None
    action_mean_bias_init: Optional[tuple[float, ...]] = None
    action_mean_head_gain: Optional[float] = None
    actor_use_layer_norm: bool = False
    value_use_layer_norm: bool = False
    actor_use_group_norm: bool = False
    value_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    value_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: Literal['mlp', 'mlp_resnet'] = 'mlp'
    critic_backbone_type: Optional[Literal['mlp', 'mlp_resnet']] = None

@dataclass
class RecurrentPPOParams(PPOParams):
    """Constructor defaults for RecurrentPPO."""

    rnn_type: Literal["lstm", "gru"] = 'lstm'
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1

@dataclass
class TransformerPPOParams(PPOParams):
    """Constructor defaults for TransformerPPO."""

    embed_dim: int = 256
    head_dim: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 3
    mlp_num: int = 2
    memory_len: int = 64
    dropout_rate: float = 0.0
    gru_bias: float = 2.0

@dataclass
class GAILParams(PPOParams):
    """Constructor defaults for GAIL."""

    n_disc_updates_per_round: int = 4
    disc_net_arch: tuple[int, ...] = (32, 32)
    disc_lr: float = 0.0003
    disc_use_adamw: bool = False
