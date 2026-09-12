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
from typing import Any, Optional

from rl_garden.common.cli_args import LoggingArgs
from rl_garden.observations import ObservationConfig


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
class D4RLLegacyConfig:
    """Legacy Gym/D4RL environment settings. CLI prefix: ``--d4rl-legacy``."""

    device: str = "cpu"


@dataclass
class MujocoConfig:
    """MuJoCo-specific env settings. CLI prefix: ``--mujoco.<field>``"""

    device: str = "cpu"
    # JSON-encoded dict forwarded verbatim to gym.make() for the task (e.g.
    # forward_reward_weight, ctrl_cost_weight, reset_noise_scale).
    env_kwargs_json: str = "{}"
    # "sync" (single process) or "async" (one OS process per env — required
    # for custom tasks with camera observations).
    vectorization: str = "sync"


@dataclass
class MujocoWarpConfig:
    """mujoco_warp GPU-specific env settings. CLI prefix: ``--mujoco_warp.<field>``"""

    device: str = "cuda:0"
    # JSON-encoded dict forwarded verbatim to the task class constructor.
    env_kwargs_json: str = "{}"


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
class CustomConfig:
    """``custom`` (PointReach template) env settings. CLI prefix: ``--custom.<field>``"""

    device: str = "cpu"


@dataclass
class RobomimicConfig:
    """robomimic-specific env settings. CLI prefix: ``--robomimic.<field>``"""

    dataset_path: Optional[str] = None
    device: str = "cpu"
    horizon: int = 400
    terminate_on_success: bool = False
    # JSON-encoded dict forwarded verbatim to the env's env_kwargs when
    # dataset_path is unset. Escape hatch, same convention as
    # ManiSkillConfig.env_kwargs_json.
    env_kwargs_json: str = "{}"


@dataclass
class OGBenchConfig:
    """OGBench-specific env settings. CLI prefix: ``--ogbench.<field>``"""

    device: str = "cpu"
    # JSON-encoded dict forwarded verbatim to gym.make() (rarely needed --
    # every OGBench task variant is already encoded in env_id itself).
    env_kwargs_json: str = "{}"
    # "sync" (single process) or "async" (one OS process per env --
    # recommended for visual-* env ids, each of which owns its own MuJoCo
    # renderer/GL context).
    vectorization: str = "sync"


@dataclass
class RLBenchConfig:
    """RLBench-specific env settings. CLI prefix: ``--rlbench.<field>``"""

    device: str = "cpu"
    headless: bool = True
    # JSON-encoded dict forwarded verbatim to rlbench.environment.Environment
    # (e.g. robot_setup, shaped_rewards, static_positions) -- rarely needed.
    env_kwargs_json: str = "{}"
    # "sync" (single process) or "async" (one OS process per env --
    # recommended once --obs.rgb/--obs.depth is set, each instance owns its
    # own CoppeliaSim renderer/GL context).
    vectorization: str = "sync"


@dataclass
class MetaWorldConfig:
    """Meta-World-specific env settings. CLI prefix: ``--metaworld.<field>``"""

    device: str = "cpu"
    # "sync" (single process) or "async" (one OS process per env).
    vectorization: str = "sync"
    # Appends a one-hot task id to the observation. Only consulted when
    # --env_id is "MT10"/"MT50".
    use_one_hot: bool = True


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
    d4rl_legacy: D4RLLegacyConfig = field(default_factory=D4RLLegacyConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    mujoco_warp: MujocoWarpConfig = field(default_factory=MujocoWarpConfig)
    isaaclab: IsaacLabConfig = field(default_factory=IsaacLabConfig)
    custom: CustomConfig = field(default_factory=CustomConfig)
    robomimic: RobomimicConfig = field(default_factory=RobomimicConfig)
    ogbench: OGBenchConfig = field(default_factory=OGBenchConfig)
    rlbench: RLBenchConfig = field(default_factory=RLBenchConfig)
    metaworld: MetaWorldConfig = field(default_factory=MetaWorldConfig)

    def resolve_backend_config(self):
        from rl_garden.envs.backend_registry import resolve_backend_config

        return resolve_backend_config(self.env_backend, self)


def make_env_request(
    args: Any,
    run_name: Optional[str] = None,
    *,
    create_eval_env: Optional[bool] = None,
) -> Any:
    """Build one ``EnvRequest`` from any training-phase args object.

    Single source of truth for the online per-algorithm
    ``_<algo>_env_request`` copies and the offline/off2on runners' inline
    ``EnvRequest(...)`` construction. ``observation`` comes from
    ``args.obs`` (every algorithm mixes in ``ObservationArgs``); an args
    object built without it (rare -- only offline evaluation-only helpers)
    has no ``obs`` attribute and gets the default (state-only)
    ``ObservationConfig()``.

    ``num_envs`` reads ``args.num_envs`` (online/off2on) and falls back to
    ``args.spec_num_envs`` (offline, which never runs a live training
    rollout -- only the eval/spec env). ``run_name`` is only needed to
    derive a video-recording directory; omit it for offline evaluation
    (which never records video and passes no ``run_name``).

    ``create_eval_env`` overrides the generic ``should_create_eval_env(args)``
    decision for entrypoints with no eval-env code path at all (e.g.
    DAgger, which is state-only by design but still carries
    ``ObservationArgs`` for CLI/config uniformity), which always pass
    ``create_eval_env=False`` explicitly.
    """
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

    num_envs = getattr(args, "num_envs", None)
    if num_envs is None:
        num_envs = args.spec_num_envs
    observation = getattr(args, "obs", None)
    if observation is None:
        observation = ObservationConfig()
    eval_record_dir = (
        resolve_eval_record_dir(args, run_name) if run_name is not None else None
    )
    return EnvRequest(
        env_id=args.env_id,
        num_envs=num_envs,
        observation=observation,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        reward_scale=getattr(args, "reward_scale", 1.0),
        reward_bias=getattr(args, "reward_bias", 0.0),
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=eval_record_dir,
        capture_video=getattr(args, "capture_video", False),
        video_fps=getattr(args, "video_fps", 30),
        num_eval_steps=args.num_eval_steps,
        create_eval_env=(
            should_create_eval_env(args) if create_eval_env is None else create_eval_env
        ),
        backend_config=args.resolve_backend_config(),
    )
