from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineUniO4Args,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class UniO4Args(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
    OfflineUniO4Args,
):
    """Uni-O4 (Unifying Online and Offline RL with Multi-Step On-Policy
    Optimization) offline pretraining -- BC-ensemble diversity + shared-
    critic PPO-clip improvement (``rl_garden/algorithms/unio4.py``).

    Box observations only. Three sequential phases share one
    ``--num_offline_steps`` budget: critic warmup
    (``--critic_warmup_steps``), BC-ensemble training
    (``--bc_ensemble_steps``), then PPO-clip improvement for whatever steps
    remain.
    """


def _unio4_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    return {
        "env": env_spec,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "critic_warmup_steps": args.critic_warmup_steps,
        "value_lr": args.value_lr,
        "q_lr": args.q_lr,
        "tau": args.tau,
        "target_update_freq": args.target_update_freq,
        "value_hidden_dims": args.value_hidden_dims,
        "q_hidden_dims": args.q_hidden_dims,
        "num_policies": args.num_policies,
        "bc_ensemble_steps": args.bc_ensemble_steps,
        "alpha_bc": args.alpha_bc,
        "actor_lr": args.actor_lr,
        "actor_hidden_dims": args.actor_hidden_dims,
        "clip_ratio": args.clip_ratio,
        "clip_decay": args.clip_decay,
        "clip_decay_steps": args.clip_decay_steps,
        "entropy_weight": args.entropy_weight,
        "omega": args.omega,
        "weight_decay": args.weight_decay,
        "use_adamw": args.use_adamw,
        "grad_clip_norm": args.grad_clip_norm,
        "use_layer_norm": args.actor_use_layer_norm,
        "use_group_norm": args.actor_use_group_norm,
        "num_groups": args.num_groups,
        "dropout_rate": args.actor_dropout_rate,
        "kernel_init": args.kernel_init,
        "backbone_type": args.backbone_type,
        "std_parameterization": args.std_parameterization,
        "seed": args.seed,
        "device": args.device,
        "logger": logger,
        "std_log": args.std_log,
        "log_freq": args.log_freq,
        "eval_env": eval_env,
        "eval_freq": args.eval_freq if eval_env is not None else 0,
        "num_eval_steps": args.num_eval_steps,
        "num_eval_episodes": args.num_eval_episodes,
        # checkpoint_dir/checkpoint_freq/save_final_checkpoint are driven by
        # run_offline_pretraining directly (see _run_offline), not the agent
        # constructor -- matches bc.py/bppo.py's convention.
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }


def build_unio4(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import UniO4
    from rl_garden.training.inspection import construct_agent

    return construct_agent(UniO4, **_unio4_kwargs(args, env_spec, logger, eval_env))


def run_unio4(args: UniO4Args) -> None:
    from rl_garden.envs.backend_registry import should_create_eval_env
    from rl_garden.training.offline._runner import run_offline

    if args.num_offline_steps <= args.critic_warmup_steps + args.bc_ensemble_steps:
        raise SystemExit(
            "--num_offline_steps "
            f"({args.num_offline_steps}) must be greater than "
            f"--critic_warmup_steps + --bc_ensemble_steps "
            f"({args.critic_warmup_steps + args.bc_ensemble_steps}), otherwise "
            "the PPO-clip improvement phase never runs and UniO4 silently "
            "degenerates into just the BC-ensemble policy."
        )
    if args.eval_freq <= 0 or args.env_id is None or not should_create_eval_env(args):
        raise SystemExit(
            "UniO4 requires periodic real-env evaluation to gate each "
            "ensemble member's old_actors sync (Phase IMPROVE's 'safe' "
            "iteration scheme) -- pass --eval_freq > 0 and a valid "
            "--env_id/env backend so an eval env is built. Without it, "
            "old_actors never advance past their BC-ensemble init and "
            "Phase IMPROVE degenerates into a single ungated PPO round per "
            "member."
        )

    run_offline(args, build_agent=build_unio4)


registry.register("unio4", UniO4Args, run_unio4)
