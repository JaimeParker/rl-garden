"""Implementation-layer config for the mujoco_warp GPU env backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MujocoWarpEnvConfig:
    env_id: str
    num_envs: int
    seed: int
    device: str = "cuda:0"
    render_width: Optional[int] = None
    render_height: Optional[int] = None
    render_rgb: bool = True
    render_depth: bool = False
    frame_stack: int = 1
    # Per-task kwargs forwarded verbatim to the task class constructor.
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_bias: float = 0.0
