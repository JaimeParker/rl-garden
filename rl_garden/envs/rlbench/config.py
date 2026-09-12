from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLBenchEnvConfig:
    task_name: str
    num_envs: int
    seed: int
    device: str = "cpu"
    # From EnvRequest.observation: which cameras render rgb / depth, whether
    # to keep the flat "state" key, and the render resolution.
    rgb_cameras: tuple[str, ...] = ()
    depth_cameras: tuple[str, ...] = ()
    state: bool = True
    image_size: tuple[int, int] = (128, 128)
    # Stacks each rgb_<cam>/depth_<cam> key into a leading time dimension via
    # ImageFrameStackWrapper (see env.py) -- only valid together with
    # rgb_cameras/depth_cameras, checked at the backend (envs/backends/
    # rlbench.py).
    frame_stack: int = 1
    headless: bool = True
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    vectorization: str = "sync"
