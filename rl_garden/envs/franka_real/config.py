"""Full env-factory config for the real-Franka env backend."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrankaRealEnvConfig:
    bridge_url: str
    device: str = "cpu"
    # (position, rotation) scale applied to the [-1, 1] action before it's
    # added as a delta to the current EE pose -- matches SERL's convention.
    action_scale: tuple[float, float] = (0.02, 0.1)
    gripper_threshold: float = 0.5
    max_episode_steps: int = 100
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Cell-specific: physical workspace bounds the EE target position is
    # clipped to before being sent to the bridge.
    safety_box_low: tuple[float, float, float] = (0.0, -0.5, 0.0)
    safety_box_high: tuple[float, float, float] = (1.0, 0.5, 0.5)
    camera_keys: tuple[str, ...] = field(default_factory=tuple)
    camera_height: int = 128
    camera_width: int = 128
