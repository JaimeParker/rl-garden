"""BCQ learning hyperparameters shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks.mlp import KernelInit
from rl_garden.networks.actor_critic import BackboneType


@dataclass
class BCQParams(OfflineOptimizationArgs):
    """Defaults of the BCQ learning algorithm."""

    # Target critic Polyak update coefficient.
    tau: float = 0.005
    # Actor learning rate.
    actor_lr: float = 1e-3
    # Critic learning rate.
    critic_lr: float = 1e-3
    # Behavior VAE learning rate.
    vae_lr: float = 1e-3
    # Hidden layer widths; None selects layers of width 400 and 300.
    net_arch: Optional[tuple[int, ...]] = None
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
    kernel_init: Optional[KernelInit] = None
    # Network backbone architecture.
    backbone_type: BackboneType = "mlp"
    # Maximum action perturbation fraction.
    phi: float = 0.05
    # Behavior VAE hidden layer width.
    vae_hidden_dim: int = 750
    # Behavior VAE latent width; None selects twice the action dimension.
    vae_latent_dim: Optional[int] = None
    # Behavior VAE KL penalty coefficient.
    beta: float = 0.5
    # Weight on the minimum in the soft double-Q target.
    soft_q_lambda: float = 0.75
