"""Learning settings for OfflineSAC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs


@dataclass
class OfflineSACParams(OfflineOptimizationArgs):
    """Constructor defaults for OfflineSAC."""

    tau: float = 0.005
    policy_lr: float = 0.0003
    q_lr: float = 0.0003
    alpha_lr: Optional[float] = None
    policy_frequency: int = 1
    target_network_frequency: int = 1
    ent_coef: float | str = 'auto'
    target_entropy: float | str = 'auto'
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = None
    n_critics: int = 2
    critic_subsample_size: Optional[int] = None
    backup_entropy: bool = True


@dataclass
class EDACParams(OfflineSACParams):
    """Offline SAC with an ensemble diversity penalty."""

    eta: float = 1.0
    n_critics: int = 10
