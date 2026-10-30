"""Opt-in fixed physical scaling; native V5 entry and defaults are unchanged."""
from dataclasses import dataclass
from typing import Literal

from .task_state_residual_sac_v5 import TaskStateResidualSACV5Args
from .task_state_residual_sac import validate_task_state_args, _task_state_env_request
from ._registry import registry


def build_scaled_task_state(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.scaled_task_state_residual import ScaledTaskStateResidualSACV5
    from .residual_sac import build_residual_sac
    validate_task_state_args(args)
    embodiment = args.robotwin.embodiment
    names = (embodiment,) if isinstance(embodiment, str) else tuple(embodiment)
    if args.state_scaling != "open_laptop_aloha_physical_v1" or names != ("aloha-agilex",):
        raise ValueError("Physical-v1 is verified only for aloha-agilex")
    return build_residual_sac(args, env, eval_env, logger, checkpoint_dir,
                              algorithm_class=ScaledTaskStateResidualSACV5)


def run_scaled_task_state(args):
    from ._runner import run_online
    validate_task_state_args(args)
    if args.base_ckpt_path is None:
        raise ValueError("An existing RGB ACT checkpoint is required")
    run_online(args, obs_tag="act_base_task_state_v5_scaled",
               make_env_request=_task_state_env_request, build_agent=build_scaled_task_state)


@dataclass
class ScaledTaskStateResidualSACV5Args(TaskStateResidualSACV5Args):
    state_scaling: Literal["open_laptop_aloha_physical_v1"] = "open_laptop_aloha_physical_v1"


registry.register("task_state_residual_sac_v5_scaled", ScaledTaskStateResidualSACV5Args,
                  run_scaled_task_state)
