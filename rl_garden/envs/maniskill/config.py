from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ManiSkillEnvConfig:
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    obs_mode: str = "state"  # "state" | "rgb" | "rgbd"
    include_state: bool = True  # only used when obs_mode is visual
    control_mode: Optional[str] = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: Optional[str] = None
    success_reward_override: Optional[float] = None
    robot_uids: Optional[str] = None
    fix_peg_pose: Optional[bool] = None
    fix_box: Optional[bool] = None
    fixed_peg_xy: Optional[tuple[float, float]] = None
    fixed_peg_z_rot_deg: Optional[float] = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    reconfiguration_freq: Optional[int] = None
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    partial_reset: bool = False
    ignore_terminations: bool = True
    record_metrics: bool = True
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # When True, keep each camera as its own ``rgb_<cam>`` / ``depth_<cam>`` key
    # instead of channel-stacking all cameras into a single ``rgb`` / ``depth``
    # tensor. Required for multi-camera envs (e.g. peg) when each camera should
    # feed an independent encoder under ``fusion_mode="per_key"``.
    per_camera_rgbd: bool = False
    # Number of visual frames exposed along a leading time dimension. Vector
    # state remains single-frame. A value of 1 disables stacking.
    frame_stack: int = 1
    # Recording
    record_dir: Optional[str] = None
    save_video: bool = False
    save_trajectory: bool = False
    max_steps_per_video: Optional[int] = None
    video_fps: int = 30
    trajectory_name: str = "trajectory"
    human_render_camera_configs: Optional[dict] = None
