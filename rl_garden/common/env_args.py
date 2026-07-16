"""Backend-agnostic CLI mixin and per-backend configuration dataclasses.

Inherit :class:`EnvBackendArgs` alongside any algorithm Args class to add
``--env_backend`` selection and per-backend sub-configs::

    @dataclass
    class SACArgs(VisionSACTrainingArgs, EnvBackendArgs):
        pass

    # CLI: python train_online.py sac --env_backend robotwin --robotwin.task_name pick_cube
    # CLI: python train_online.py sac --maniskill.sim-backend physx_cpu
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rl_garden.common.cli_args import LoggingArgs


@dataclass
class EnvRunArgs(LoggingArgs):
    """Environment runtime fields shared by online and off2on training."""

    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    seed: int = 1
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    capture_video: bool = True
    video_fps: int = 30
    eval_output_dir: Optional[str] = None


@dataclass
class ManiSkillConfig:
    """ManiSkill-specific env settings. CLI prefix: ``--maniskill.<field>``"""

    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    reward_mode: Optional[str] = None
    # JSON-encoded dict forwarded verbatim to ManiSkillEnvConfig.env_kwargs, which
    # takes precedence over any named field there. Escape hatch for task-specific
    # kwargs (e.g. peg overrides) without adding named fields here per task.
    env_kwargs_json: str = "{}"


@dataclass
class RoboTwinConfig:
    """RoboTwin-specific env settings. CLI prefix: ``--robotwin.<field>``"""

    # camera / rendering
    include_wrist_cameras: bool = True
    random_light: bool = False
    crazy_random_light_rate: float = 0.0
    head_camera_type: str = "D435"
    wrist_camera_type: str = "D435"
    render_every_control_step: bool = False
    control_step_cap: Optional[int] = None
    profile_timing: bool = False
    profile_interval: int = 100

    # task / planner
    robotwin_root: Optional[str] = None
    assets_path: Optional[str] = None
    seeds_path: Optional[str] = None
    step_lim: int = 400
    planner_backend: str = "mplib"
    embodiment: list = field(default_factory=lambda: ["aloha-agilex"])
    reward_mode: str = "dense"
    disable_topp: bool = False

    # control scaling
    joint_delta_scale: float = 0.05
    gripper_delta_scale: float = 0.2
    ee_delta_pos_scale: float = 0.03
    ee_delta_rot_scale: float = 0.15

    # device
    device: str = "auto"


@dataclass
class MinariConfig:
    """Minari-specific env settings. CLI prefix: ``--minari.<field>``"""

    device: str = "cpu"
    download: bool = True


@dataclass
class IsaacLabConfig:
    """IsaacLab-specific env settings. CLI prefix: ``--isaaclab.<field>``"""

    headless: bool = True
    sim_device: str = "cuda:0"
    # JSON-encoded dict forwarded verbatim to IsaacLabEnvConfig.env_kwargs
    # (e.g. task-specific cfg overrides). Escape hatch, same convention as
    # ManiSkillConfig.env_kwargs_json.
    env_kwargs_json: str = "{}"


@dataclass
class EnvBackendArgs:
    """Mixin: adds ``env_backend`` selector and per-backend sub-configs.

    All fields have defaults so this can be appended to any dataclass
    inheritance chain without requiring positional-argument changes.
    """

    env_backend: str = "maniskill"
    maniskill: ManiSkillConfig = field(default_factory=ManiSkillConfig)
    robotwin: RoboTwinConfig = field(default_factory=RoboTwinConfig)
    minari: MinariConfig = field(default_factory=MinariConfig)
    isaaclab: IsaacLabConfig = field(default_factory=IsaacLabConfig)

    def resolve_backend_config(self):
        from rl_garden.envs.backend_registry import resolve_backend_config

        return resolve_backend_config(self.env_backend, self)
