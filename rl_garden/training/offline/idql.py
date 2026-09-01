from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineCommonArgs,
    OfflineCriticArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineIDQLArgs,
    OfflineValueArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class IDQLArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineCriticArgs,
    OfflineValueArgs,
    OfflineIDQLArgs,
):
    """IDQL: IQL-style expectile value/critic regression + diffusion actor."""


def _idql_kwargs(args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None) -> dict:
    return {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "tau": args.tau,
        "offline_sampling": args.offline_sampling,
        "actor_tau": args.actor_tau,
        "actor_objective": args.actor_objective,
        "policy_temperature": args.policy_temperature,
        "expectile": args.expectile,
        "critic_value_lr": args.critic_value_lr,
        "actor_lr": args.actor_lr,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "lr_schedule": args.lr_schedule,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
        "lr_min_ratio": args.lr_min_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "n_critics": args.n_critics,
        "critic_subsample_size": args.critic_subsample_size,
        "critic_use_layer_norm": args.critic_use_layer_norm,
        "value_use_layer_norm": args.value_use_layer_norm,
        "diffusion_mlp_dims": args.diffusion_mlp_dims,
        "denoising_steps": args.denoising_steps,
        "schedule": args.schedule,
        "n_action_samples": args.n_action_samples,
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


def build_idql(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import IDQL
    from rl_garden.training.inspection import construct_agent

    return construct_agent(IDQL, **_idql_kwargs(args, env_spec, logger, eval_env))


def run_idql(args: IDQLArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_idql)


registry.register("idql", IDQLArgs, run_idql)
