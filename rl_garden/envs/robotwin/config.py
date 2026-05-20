"""Configuration for RoboTwin environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


ControlMode = Literal["delta_joint_pos", "joint_pos"]
RewardMode = Literal["dense", "sparse"]


@dataclass
class RoboTwinEnvConfig:
    """Settings for :func:`make_robotwin_env`.

    RoboTwin itself is treated as an optional runtime dependency. ``robotwin_root``
    can point at a cloned RoboTwin repository when it is not already importable.
    """

    task_name: str = "place_shoe"
    num_envs: int = 1
    seed: int = 0
    device: str = "auto"

    # RoboTwin runtime paths/config.
    robotwin_root: Optional[str] = None
    assets_path: Optional[str] = None
    seeds_path: Optional[str] = None
    task_config: dict[str, Any] = field(default_factory=dict)

    # Episode/reset behavior.
    auto_reset: bool = True
    ignore_terminations: bool = False
    max_episode_steps: Optional[int] = None
    step_lim: Optional[int] = None
    group_size: int = 1
    use_fixed_reset_state_ids: bool = False
    record_metrics: bool = True

    # Observation/action behavior.
    include_wrist_cameras: bool = True
    image_size: tuple[int, int] = (224, 224)
    control_mode: ControlMode = "delta_joint_pos"
    action_dim: int = 14
    joint_delta_scale: float = 0.05
    gripper_delta_scale: float = 0.2

    # Reward behavior.
    reward_mode: RewardMode = "dense"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    use_relative_reward: bool = False

    # RoboTwin task defaults copied from RLinf_support env configs where useful.
    planner_backend: str = "mplib"
    embodiment: list[str] = field(default_factory=lambda: ["aloha-agilex"])
    instruction_type: str = "seen"
    clear_cache_freq: int = 8
