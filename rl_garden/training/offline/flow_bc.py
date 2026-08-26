from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineFlowBCArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class FlowBCArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineFlowBCArgs,
):
    """Flow-matching behavior cloning offline pretraining."""


def _flow_bc_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from gymnasium import spaces

    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.encoders import discover_image_keys

    obs_space = env_spec.single_observation_space
    kwargs = {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "offline_sampling": args.offline_sampling,
        "actor_lr": args.actor_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "net_arch": args.net_arch,
        "flow_steps": args.flow_steps,
        "actor_use_layer_norm": args.actor_use_layer_norm,
        "kernel_init": args.kernel_init,
        "activation_fn": args.activation_fn,
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "eval_env": eval_env,
        "eval_freq": args.eval_freq if eval_env is not None else 0,
        "num_eval_steps": args.num_eval_steps,
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }
    if isinstance(obs_space, spaces.Dict):
        image_keys = discover_image_keys(obs_space)
        kwargs.update(
            image_encoder_factory=image_encoder_factory_from_args(args),
            image_keys=image_keys,
            state_key="state",
            use_proprio=args.include_state,
            image_fusion_mode=args.image_fusion_mode,
            enable_stacking=False,
        )
    return kwargs


def build_flow_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FlowBC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FlowBC, **_flow_bc_kwargs(args, env_spec, logger, eval_env))


def run_flow_bc(args: FlowBCArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_flow_bc)


registry.register("flow_bc", FlowBCArgs, run_flow_bc)
