"""IDQL learning hyperparameters shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs


@dataclass
class IDQLParams(OfflineOptimizationArgs):
    """Defaults of the IDQL learning algorithm."""

    # Expectile of the value regression.
    expectile: float = 0.7
    # Target critic Polyak update coefficient.
    tau: float = 0.005
    # Target diffusion actor Polyak update coefficient.
    actor_tau: float = 1e-3
    # Diffusion actor behavior-cloning or advantage-weighted objective.
    actor_objective: Literal["bc", "soft_adv", "hard_adv", "exp_adv"] = "bc"
    # Multiplier in exponentiated actor advantage weights.
    policy_temperature: float = 3.0
    # Critic hidden layer widths.
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    # Value-network hidden layer widths.
    value_hidden_dims: tuple[int, ...] = (256, 256)
    # Number of critic networks.
    n_critics: int = 2
    # Number of target critics sampled; None uses the full ensemble.
    critic_subsample_size: Optional[int] = None
    # Apply layer normalization in the critic.
    critic_use_layer_norm: bool = False
    # Apply layer normalization in the value network.
    value_use_layer_norm: bool = False
    # Diffusion actor hidden layer widths.
    diffusion_mlp_dims: tuple[int, ...] = (256, 256)
    # Number of diffusion denoising steps.
    denoising_steps: int = 5
    # Diffusion noise schedule.
    schedule: Literal["cosine", "vp", "linear"] = "vp"
    # Number of diffusion action candidates.
    n_action_samples: int = 64
    # Joint critic and value learning rate.
    critic_value_lr: float = 3e-4
    # Actor learning rate.
    actor_lr: float = 3e-4
