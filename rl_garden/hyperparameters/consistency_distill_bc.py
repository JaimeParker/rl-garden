"""ConsistencyDistillBC learning hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rl_garden.hyperparameters.diffusion_bc import DiffusionPolicyParams


@dataclass
class ConsistencyDistillBCParams(DiffusionPolicyParams):
    """Constructor defaults for ConsistencyDistillBC."""

    cm_lr: float = 0.0001
    weight_decay: float = 1e-06
    cm_ema_decay: float = 0.95
    cm_grad_clip_norm: Optional[float] = 1.0
    cm_sigma_data: float = 0.5
    cm_timestep_scaling: float = 0.1
