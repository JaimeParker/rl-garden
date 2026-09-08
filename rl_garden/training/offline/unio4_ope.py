from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineUniO4OPEArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class UniO4OPEArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
    OfflineUniO4OPEArgs,
):
    """UniO4OPE: Uni-O4 with dynamics-model OPE gating -- a faithful-
    reproduction variant of ``unio4`` (``rl_garden/algorithms/unio4_ope.py``).
    Real-env eval keeps running for monitoring (same as ``unio4``'s), but
    ``old_actors`` sync is gated by a dynamics-model rollout Q-estimate
    instead, matching upstream Uni-O4's own ``ope_dynamics_eval``. Box
    observations only, same three-phase budget as ``unio4``.
    """


def _unio4_ope_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    return {
        "env": env_spec,
        "task": args.env_id,
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
        "dynamics_hidden_dims": args.dynamics_hidden_dims,
        "dynamics_n_ensemble": args.dynamics_n_ensemble,
        "dynamics_n_elites": args.dynamics_n_elites,
        "dynamics_lr": args.dynamics_lr,
        "dynamics_weight_decay": args.dynamics_weight_decay,
        "dynamics_max_epochs_since_update": args.dynamics_max_epochs_since_update,
        "dynamics_max_epochs": args.dynamics_max_epochs,
        "dynamics_batch_size": args.dynamics_batch_size,
        "dynamics_holdout_ratio": args.dynamics_holdout_ratio,
        "ope_rollout_length": args.ope_rollout_length,
        "ope_rollout_batch_size": args.ope_rollout_batch_size,
        "ope_gating_freq": args.ope_gating_freq,
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
        # constructor -- matches bc.py/bppo.py/unio4.py's convention.
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }


def build_unio4_ope(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import UniO4OPE
    from rl_garden.training.inspection import construct_agent

    return construct_agent(UniO4OPE, **_unio4_ope_kwargs(args, env_spec, logger, eval_env))


def run_unio4_ope(args: UniO4OPEArgs) -> None:
    from rl_garden.envs.backend_registry import should_create_eval_env
    from rl_garden.training.offline._runner import run_offline

    if args.num_offline_steps <= args.critic_warmup_steps + args.bc_ensemble_steps:
        raise SystemExit(
            "--num_offline_steps "
            f"({args.num_offline_steps}) must be greater than "
            f"--critic_warmup_steps + --bc_ensemble_steps "
            f"({args.critic_warmup_steps + args.bc_ensemble_steps}), otherwise "
            "the PPO-clip improvement phase never runs and old_actors never "
            "advances past its BC-ensemble init."
        )
    if args.env_id is None:
        raise SystemExit(
            "UniO4OPE requires --env_id: it picks a termination_fn for the "
            "dynamics-model rollout from it (halfcheetah/hopper/walker2d), "
            "unlike plain unio4 which doesn't need a task string."
        )
    if args.eval_freq <= 0 or not should_create_eval_env(args):
        raise SystemExit(
            "UniO4OPE still requires periodic real-env evaluation for "
            "monitoring (matches upstream's off_evaluate) even though it no "
            "longer gates old_actors sync -- pass --eval_freq > 0 and a "
            "valid env backend so an eval env is built."
        )

    run_offline(args, build_agent=build_unio4_ope)


registry.register("unio4_ope", UniO4OPEArgs, run_unio4_ope)
