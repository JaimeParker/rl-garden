from __future__ import annotations

from dataclasses import dataclass

from rl_garden.hyperparameters import params_kwargs
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.training.offline.args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineUniO4OPEArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class UniO4OPEArgs(
    OfflineUniO4OPEArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
):
    """UniO4OPE: Uni-O4 with dynamics-model OPE gating -- a faithful-
    reproduction variant of ``unio4`` (``rl_garden/algorithms/unio4_ope.py``).
    Real-env eval keeps running for monitoring (same as ``unio4``'s), but
    ``old_actors`` sync is gated by a dynamics-model rollout Q-estimate
    instead, matching upstream Uni-O4's own ``ope_dynamics_eval``. Box
    observations only, same three-phase budget as ``unio4``.
    """

    # None uses the older --actor-use-layer-norm flag.
    use_layer_norm: Optional[bool] = None

    # None uses the older --actor-use-group-norm flag.
    use_group_norm: Optional[bool] = None

    # None uses the older --actor-dropout-rate flag.
    dropout_rate: Optional[float] = None


def _unio4_ope_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.algorithms import UniO4OPE

    return {
        **params_kwargs(args, UniO4OPE.params_class),
        "env": env_spec,
        "task": args.env_id,
        "buffer_size": args.buffer_size,
        "buffer_device": args.buffer_device,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "use_layer_norm": (
            args.actor_use_layer_norm if args.use_layer_norm is None else args.use_layer_norm
        ),
        "use_group_norm": (
            args.actor_use_group_norm if args.use_group_norm is None else args.use_group_norm
        ),
        "dropout_rate": (
            args.actor_dropout_rate if args.dropout_rate is None else args.dropout_rate
        ),
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
    from rl_garden.training.offline.runner import run_offline

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


def _unio4_ope_algorithm_cls() -> type:
    from rl_garden.algorithms import UniO4OPE

    return UniO4OPE


registry.register(
    "unio4_ope",
    UniO4OPEArgs,
    run_unio4_ope,
    algorithm_cls=_unio4_ope_algorithm_cls,
    build_agent=build_unio4_ope,
)
