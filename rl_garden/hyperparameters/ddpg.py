"""DDPG learning settings shared by constructors and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs


@dataclass
class DDPGParams(OfflineOptimizationArgs):
    """Defaults matching DDPG construction."""

    tau: float = 0.01
    policy_lr: float = 0.0001
    q_lr: float = 0.0001
    feature_dim: int = 50
    hidden_dim: int = 1024
    nstep: int = 3
    stddev_schedule: str = 'linear(1.0,0.1,500000)'
    stddev_clip: float = 0.3
    num_expl_steps: int = 2000
