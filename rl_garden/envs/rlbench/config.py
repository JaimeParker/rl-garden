from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_garden.buffers.rlbench_dataset import RLBENCH_CAMERA_NAMES


@dataclass
class RLBenchEnvConfig:
    task_name: str
    num_envs: int
    seed: int
    device: str = "cpu"
    obs_mode: str = "state"
    cameras: tuple[str, ...] = RLBENCH_CAMERA_NAMES
    image_size: tuple[int, int] = (128, 128)
    headless: bool = True
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    vectorization: str = "sync"
