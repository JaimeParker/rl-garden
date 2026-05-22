"""Configuration for RoboTwin environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


ControlMode = Literal["delta_joint_pos", "joint_pos", "ee_delta_pose"]
RewardMode = Literal["dense", "sparse"]
ExecutorType = Literal["thread", "process"]


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
    random_light: bool = False
    crazy_random_light_rate: float = 0.0

    # Observation/action behavior.
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
    use_relative_reward: bool = False

    # RoboTwin task defaults copied from RLinf_support env configs where useful.
    planner_backend: str = "mplib"
    embodiment: list[str] = field(default_factory=lambda: ["aloha-agilex"])
    instruction_type: str = "seen"
    clear_cache_freq: int = 8

    # Executor backend: "thread" (default) uses ThreadPoolExecutor; "process"
    # spawns one OS process per env so mplib TOPP planners run on separate CPU
    # cores without contention.
    executor_type: ExecutorType = "thread"
    # Pin each worker process to a dedicated CPU core (process executor only).
    cpu_affinity: bool = False

    # Offload mplib TOPP planning to N dedicated worker processes (one per env)
    # so TOPP calls from different threads run truly in parallel without GIL
    # contention.  SAPIEN stays in the main process — no GPU OOM.
    parallel_topp: bool = False
    # Pin each TOPP worker process to a dedicated CPU core.
    topp_cpu_affinity: bool = False
    # Max envs allowed to run scene.step() ctrl loop simultaneously.
    # 0 = auto (1 when parallel_topp=True, unlimited otherwise).
    # 1 = fully serial (eliminates GPU thundering herd with parallel_topp).
    # N = allow N concurrent ctrl loops (experiment with intermediate values).
    ctrl_concurrency: int = 0

    def __post_init__(self) -> None:
        if self.control_mode == "ee_delta_pose" and self.action_dim != 14:
            raise ValueError("ee_delta_pose control mode requires action_dim=14.")
