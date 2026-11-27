"""UniO4OPE learning parameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


from rl_garden.hyperparameters.unio4 import UniO4Params


@dataclass
class UniO4OPEParams(UniO4Params):
    """Dynamics training and model-based policy gating settings."""

    dynamics_hidden_dims: tuple[int, ...] = (200, 200, 200, 200)
    dynamics_n_ensemble: int = 7
    dynamics_n_elites: int = 5
    dynamics_lr: float = 0.001
    dynamics_weight_decay: tuple[float, ...] = (2.5e-05, 5e-05, 7.5e-05, 7.5e-05, 0.0001)
    dynamics_max_epochs_since_update: int = 5
    dynamics_max_epochs: Optional[int] = None
    dynamics_batch_size: int = 256
    dynamics_holdout_ratio: float = 0.2
    ope_rollout_length: int = 1000
    ope_rollout_batch_size: int = 512
    ope_gating_freq: int = 100
