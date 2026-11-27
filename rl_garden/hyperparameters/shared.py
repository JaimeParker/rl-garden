"""Shared optimization, network, replay and update-loop parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class OfflineOptimizationArgs:
    # Optimizer weight decay.
    weight_decay: float = 0.0
    # Use decoupled AdamW weight decay.
    use_adamw: bool = False
    # Learning-rate schedule.
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    # Number of learning-rate warmup updates.
    lr_warmup_steps: int = 0
    # Number of learning-rate decay updates.
    lr_decay_steps: int = 0
    # Minimum learning-rate fraction.
    lr_min_ratio: float = 0.0
    # Maximum gradient norm, or None to disable clipping.
    grad_clip_norm: Optional[float] = None


@dataclass
class OfflineDeterministicActorCriticArgs:
    """Network toggles shared by TD3-BC and AWAC (2 fixed critics, no ensemble
    subsampling -- unlike ``OfflineActorArgs``/``OfflineCriticArgs``, which are
    tuned for CQL/IQL's larger critic ensembles)."""

    # Apply layer normalization in the actor.
    actor_use_layer_norm: bool = False
    # Apply layer normalization in the critic.
    critic_use_layer_norm: bool = False
    # Apply group normalization in the actor.
    actor_use_group_norm: bool = False
    # Apply group normalization in the critic.
    critic_use_group_norm: bool = False
    # Number of normalization groups.
    num_groups: int = 32
    # Actor dropout probability, or None to disable.
    actor_dropout_rate: Optional[float] = None
    # Critic dropout probability, or None to disable.
    critic_dropout_rate: Optional[float] = None
    # Network weight initialization.
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    # Network backbone architecture.
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    # Number of critic networks.
    n_critics: int = 2


@dataclass
class OfflineCriticArgs:
    n_critics: int = 10
    critic_subsample_size: int = 2
    critic_use_layer_norm: bool = True
    critic_use_group_norm: bool = False
    critic_dropout_rate: Optional[float] = None


@dataclass
class OfflineActorArgs:
    actor_use_layer_norm: bool = True
    actor_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    kernel_init: Optional[
        Literal[
            "xavier_uniform",
            "xavier_normal",
            "orthogonal",
            "kaiming_uniform",
            "orthogonal_near_zero_output",
        ]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    std_parameterization: Literal["exp", "uniform"] = "exp"


@dataclass
class OfflineReplayArgs:
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    offline_sampling: Literal["with_replace", "without_replace"] = "with_replace"


@dataclass
class OfflineDiscountArgs:
    gamma: float = 0.99
    tau: float = 0.005
    utd: float = 1.0


@dataclass
class Off2OnLoopArgs(OfflineReplayArgs, OfflineDiscountArgs):
    """Replay and update-loop hyperparameters of the off2on family."""

    learning_starts: int = 4_000
    training_freq: int = 64
    # Offline-to-online family update-to-data ratio.
    utd: float = 4.0
