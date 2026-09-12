from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineMeanFlowBCArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class MeanFlowBCArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineMeanFlowBCArgs,
):
    """MeanFlow behavior cloning offline pretraining."""


def _mean_flow_bc_kwargs(
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
        "num_sample_steps": args.num_sample_steps,
        "mode": args.mode,
        "time_dist_mu": args.time_dist_mu,
        "time_dist_sigma": args.time_dist_sigma,
        "adaptive_l2_gamma": args.adaptive_l2_gamma,
        "adaptive_l2_c": args.adaptive_l2_c,
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
        # No critic here, so encoder_sharing is irrelevant (see algorithm
        # module docstring); no critic_encoder_config either.
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "image_augmentation_seed": args.seed,
    }
    return kwargs


def build_mean_flow_bc(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import MeanFlowBC
    from rl_garden.training.inspection import construct_agent

    return construct_agent(
        MeanFlowBC, **_mean_flow_bc_kwargs(args, env_spec, logger, eval_env)
    )


def run_mean_flow_bc(args: MeanFlowBCArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_mean_flow_bc)


registry.register("mean_flow_bc", MeanFlowBCArgs, run_mean_flow_bc)
