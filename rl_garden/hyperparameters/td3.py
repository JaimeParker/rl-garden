"""TD3 learning settings shared by constructors and CLI Args."""
from __future__ import annotations

from dataclasses import dataclass

from rl_garden.hyperparameters.ddpg import DDPGParams


@dataclass
class TD3Params(DDPGParams):
    """Defaults matching TD3 construction."""

    policy_freq: int = 2
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
