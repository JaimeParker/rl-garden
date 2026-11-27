from __future__ import annotations

from dataclasses import dataclass

from rl_garden.hyperparameters import params_kwargs
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.training.offline.args import (
    OfflineActorArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineUniO4Args,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class UniO4Args(
    OfflineUniO4Args,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
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

    # None uses the older --actor-use-layer-norm flag.
    use_layer_norm: Optional[bool] = None

    # None uses the older --actor-use-group-norm flag.
    use_group_norm: Optional[bool] = None

    # None uses the older --actor-dropout-rate flag.
    dropout_rate: Optional[float] = None


def _unio4_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.algorithms import UniO4

    return {
        **params_kwargs(args, UniO4.params_class),
        "env": env_spec,
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
    from rl_garden.training.offline.runner import run_offline

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




def _uni_o4_algorithm_cls() -> type:
    from rl_garden.algorithms import UniO4

    return UniO4

registry.register(
    "unio4", UniO4Args, run_unio4, algorithm_cls=_uni_o4_algorithm_cls, build_agent=build_unio4
)
