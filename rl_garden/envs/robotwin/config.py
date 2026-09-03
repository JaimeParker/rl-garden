"""Configuration for RoboTwin environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


ControlMode = Literal["delta_joint_pos", "joint_pos", "delta_ee"]
RewardMode = Literal["dense", "sparse"]
RewardShapingMode = Literal["absolute", "relative", "potential", "hybrid"]


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

    # Profiling.
    profile_timing: bool = False
    profile_interval: int = 50

    # RoboTwin runtime performance knobs.
    render_every_control_step: bool = False
    control_step_cap: Optional[int] = None
    random_background: bool = True
    cluttered_table: bool = True
    clean_background_rate: float = 0.02
    random_head_camera_dis: float = 0.0
    random_table_height: float = 0.03
    random_light: bool = False
    crazy_random_light_rate: float = 0.0

    # Observation/action behavior. RoboTwin-native end-effector delta control is
    # named "delta_ee" to match RoboTwin's take_action(action_type=...).
    include_wrist_cameras: bool = True
    image_size: tuple[int, int] = (224, 224)
    head_camera_type: str = "D435"
    wrist_camera_type: str = "D435"
    control_mode: ControlMode = "delta_joint_pos"
    action_dim: int = 14
    joint_delta_scale: float = 0.05
    ee_delta_pos_scale: float = 0.03
    ee_delta_rot_scale: float = 0.15
    gripper_delta_scale: float = 0.2

    # Reward behavior.
    reward_mode: RewardMode = "dense"
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    reward_shaping_mode: RewardShapingMode = "absolute"
    use_relative_reward: bool = False
    dense_success_reward: float = 1.0
    potential_discount: float = 0.99
    potential_weight: float = 5.0
    dense_weight: float = 0.03
    relative_weight: float = 3.0
    step_penalty: float = 0.003
    stall_threshold: float = 1e-4
    stall_penalty: float = 0.035
    backtrack_penalty: float = 0.06

    # RoboTwin task defaults copied from RLinf_support env configs where useful.
    planner_backend: str = "mplib"
    embodiment: list[str] = field(default_factory=lambda: ["aloha-agilex"])
    instruction_type: str = "seen"
    clear_cache_freq: int = 8

    def __post_init__(self) -> None:
        if self.control_mode == "delta_ee" and self.action_dim != 14:
            raise ValueError("delta_ee control mode requires action_dim=14.")
        if self.reward_mode not in {"dense", "sparse"}:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode!r}; "
                "expected 'dense' or 'sparse'."
            )
        if self.reward_shaping_mode not in {
            "absolute", "relative", "potential", "hybrid"
        }:
            raise ValueError(
                f"Unsupported reward_shaping_mode={self.reward_shaping_mode!r}."
            )
        if self.use_relative_reward and self.reward_shaping_mode != "absolute":
            raise ValueError(
                "use_relative_reward is the legacy dense reward delta switch; "
                "do not combine it with reward_shaping_mode."
            )
        nonnegative = {
            "dense_success_reward": self.dense_success_reward,
            "potential_weight": self.potential_weight,
            "dense_weight": self.dense_weight,
            "relative_weight": self.relative_weight,
            "step_penalty": self.step_penalty,
            "stall_threshold": self.stall_threshold,
            "stall_penalty": self.stall_penalty,
            "backtrack_penalty": self.backtrack_penalty,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if not 0.0 <= self.potential_discount <= 1.0:
            raise ValueError("potential_discount must be in [0, 1].")
