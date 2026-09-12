from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rl_garden.observations import ObservationConfig


@dataclass
class ManiSkillEnvConfig:
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    # From EnvRequest.observation: which cameras (by ManiSkill's own sensor
    # names, e.g. "base_camera", "hand_camera") render rgb / depth, whether
    # to keep the flat "state" key, and the render resolution. Always
    # produces per-camera ``rgb_<cam>``/``depth_<cam>`` keys -- no
    # channel-stacked bare ``rgb``/``depth``.
    rgb_cameras: tuple[str, ...] = ()
    depth_cameras: tuple[str, ...] = ()
    state: bool = True
    control_mode: Optional[str] = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: Optional[str] = None
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

    @classmethod
    def from_observation(
        cls, obs: "ObservationConfig", **overrides: Any
    ) -> "ManiSkillEnvConfig":
        """Build from the generic ``ObservationConfig`` -- the only place that
        translates ``image_size`` (H, W) into ManiSkill's own per-sensor
        ``camera_width``/``camera_height`` fields.
        """
        height, width = obs.image_size if obs.image_size is not None else (None, None)
        return cls(
            rgb_cameras=obs.rgb,
            depth_cameras=obs.depth,
            state=obs.state,
            frame_stack=obs.frame_stack,
            camera_width=width,
            camera_height=height,
            **overrides,
        )
