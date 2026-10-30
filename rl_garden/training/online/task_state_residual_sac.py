"""Explicit open_laptop state-v2 residual entry; never changes legacy defaults."""
from __future__ import annotations


def validate_task_state_args(args):
    if (args.env_backend, args.env_id, args.control_mode) != ("robotwin", "open_laptop", "delta_ee"):
        raise ValueError("Task state v2 requires robotwin/open_laptop/delta_ee")
    if args.obs_mode != "state" or not args.include_state or args.frame_stack != 1:
        raise ValueError("Task state v2 requires unstacked state observations")
    if tuple(args.robotwin.extra_state) != (f"open_laptop_robot_v{args.STATE_VERSION}",):
        raise ValueError("Select exactly the matching task state version")
    if args.base_policy != "act" or args.debug or args.num_envs != 1:
        raise ValueError("This validated entry requires frozen ACT and one environment")
    if args.image_augmentation != "none" or args.image_keys is not None:
        raise ValueError("Residual state v2 uses no image keys or image augmentation")
    if args.offline_dataset_path is not None:
        raise ValueError("Offline replay mixing is not validated for task state v2")


def _task_state_env_request(args, run_name):
    from rl_garden.training.online.residual_sac import _residual_sac_env_request

    validate_task_state_args(args)
    return _residual_sac_env_request(args, run_name)


def build_task_state_residual_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.task_state_residual import TaskStateResidualSACV2, TaskStateResidualSACV3, TaskStateResidualSACV4, TaskStateResidualSACV5
    from rl_garden.training.online.residual_sac import build_residual_sac

    validate_task_state_args(args)
    cls = {2: TaskStateResidualSACV2, 3: TaskStateResidualSACV3, 4: TaskStateResidualSACV4, 5: TaskStateResidualSACV5}[args.STATE_VERSION]
    return build_residual_sac(args, env, eval_env, logger, checkpoint_dir, algorithm_class=cls)


def run_task_state_residual_sac(args):
    from rl_garden.training.online._runner import run_online

    validate_task_state_args(args)
    if args.base_ckpt_path is None:
        raise ValueError("An existing RGB ACT checkpoint is required")
    run_online(args, obs_tag=f"act_base_task_state_v{args.STATE_VERSION}",
               make_env_request=_task_state_env_request,
               build_agent=build_task_state_residual_sac)


from dataclasses import dataclass, field  # noqa: E402
from typing import ClassVar, Optional  # noqa: E402
from rl_garden.common.env_args import RoboTwinConfig  # noqa: E402
from rl_garden.training.online.residual_sac import ResidualSACArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class TaskStateResidualSACArgs(ResidualSACArgs):
    """114D task-state residual with an RGB ACT base policy."""

    STATE_VERSION: ClassVar[int] = 2

    env_backend: str = "robotwin"
    env_id: str = "open_laptop"
    control_mode: str = "delta_ee"
    obs_mode: str = "state"
    num_envs: int = 1
    num_eval_envs: int = 0
    image_keys: Optional[str] = None
    base_ckpt_path: Optional[str] = None
    base_act_observation_width: Optional[int] = 320
    base_act_observation_height: Optional[int] = 240
    robotwin: RoboTwinConfig = field(default_factory=lambda: RoboTwinConfig(
        extra_state=("open_laptop_robot_v2",)))

    def __post_init__(self):
        validate_task_state_args(self)


registry.register("task_state_residual_sac", TaskStateResidualSACArgs,
                  run_task_state_residual_sac)
