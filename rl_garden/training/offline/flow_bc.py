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
    from rl_garden.common.cli_args import resolve_obs_groups_config
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
        # Passed unconditionally rather than gated on --obs.is_visual: the
        # dataset's own stored keys decide state-vs-Dict shape, not --obs
        # (see infer_specs_from_h5), so --obs.is_visual can't be trusted
        # here. A state-only dataset combined with a non-default
        # image-related --encoder.* field now raises ObservationContractError
        # (build_observation_encoder) instead of silently ignoring it.
        "encoder_config": args.encoder,
        "obs_groups": resolve_obs_groups_config(args),
    }
    return kwargs


def build_flow_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FlowBC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FlowBC, **_flow_bc_kwargs(args, env_spec, logger, eval_env))


def run_flow_bc(args: FlowBCArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_flow_bc)


registry.register("flow_bc", FlowBCArgs, run_flow_bc)
