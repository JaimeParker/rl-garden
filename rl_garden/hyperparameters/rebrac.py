"""ReBRAC learning hyperparameters, shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from rl_garden.hyperparameters.td3_bc import TD3CoreParams


@dataclass
class ReBRACParams(TD3CoreParams):
    """Defaults matching the ReBRAC learning core."""

    # Actor learning rate.
    actor_lr: float = 1e-3

    # Critic learning rate.
    critic_lr: float = 1e-3

    # Actor behavior-cloning penalty coefficient.
    actor_bc_coef: float = 1.0

    # Critic bootstrap behavior-cloning penalty coefficient.
    critic_bc_coef: float = 1.0

    # Normalize the actor Q objective by its absolute mean.
    normalize_q: bool = True

    # Apply layer normalization in the critic.
    critic_use_layer_norm: bool = True
