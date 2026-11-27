"""Learning settings for FlashSAC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FlashSACParams:
    """Constructor defaults for FlashSAC."""

    tau: float = 0.005
    n_step: int = 3
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    num_bins: int = 101
    min_v: float = -5.0
    max_v: float = 5.0
    actor_lr: float = 0.0001
    critic_lr: float = 0.0001
    alpha_lr: float = 0.0001
    actor_update_period: int = 1
    grad_clip_norm: Optional[float] = None
    temp_initial_value: float = 0.01
    target_entropy: float | str = 'auto'
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    normalize_reward: bool = False
    normalized_g_max: float = 10.0
    bc_alpha: float = 0.0
