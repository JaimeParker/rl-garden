from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RobomimicEnvConfig:
    env_id: str
    num_envs: int = 1
    dataset_path: Optional[str] = None
    device: str = "cpu"
    horizon: int = 400
    terminate_on_success: bool = False
    env_kwargs_json: str = "{}"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
