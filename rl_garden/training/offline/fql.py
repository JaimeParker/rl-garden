"""FQL offline pretraining registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gymnasium import spaces

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineFQLArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class FQLArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineFQLArgs,
):
    """FQL offline pretraining. Box or Dict (vision) observations."""


def _fql_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    kwargs = {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "offline_sampling": args.offline_sampling,
        "tau": args.tau,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "alpha": args.alpha,
        "flow_steps": args.flow_steps,
        "q_agg": args.q_agg,
        "normalize_q_loss": args.normalize_q_loss,
        "n_critics": args.n_critics,
        "actor_use_layer_norm": args.actor_use_layer_norm,
        "critic_use_layer_norm": args.critic_use_layer_norm,
        "actor_use_group_norm": args.actor_use_group_norm,
        "critic_use_group_norm": args.critic_use_group_norm,
        "num_groups": args.num_groups,
        "critic_dropout_rate": args.critic_dropout_rate,
        "kernel_init": args.kernel_init,
        "backbone_type": args.backbone_type,
        "activation_fn": args.activation_fn,
        "encoder_sharing": args.encoder_sharing,
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
    if isinstance(env_spec.single_observation_space, spaces.Dict):
        from rl_garden.common.cli_args import image_encoder_factory_from_args

        kwargs["image_encoder_factory"] = image_encoder_factory_from_args(args)
    return kwargs


def build_fql(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import FQL
    from rl_garden.training.inspection import construct_agent

    return construct_agent(FQL, **_fql_kwargs(args, env_spec, logger, eval_env))


def run_fql(args: FQLArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_fql)


registry.register("fql", FQLArgs, run_fql)
