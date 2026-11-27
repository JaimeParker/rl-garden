from __future__ import annotations

from dataclasses import dataclass

from rl_garden.hyperparameters import params_kwargs
from typing import TYPE_CHECKING, Any, Optional

from rl_garden.training.offline.args import (
    OfflineActorArgs,
    OfflineBPPOArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
)
from rl_garden.training.offline.registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class BPPOArgs(
    OfflineBPPOArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
):
    """BPPO (Behavior Proximal Policy Optimization) offline pretraining.

    Box observations only. Pair with ``--bc_checkpoint`` (a ``bc``
    pretrained checkpoint) to warm-start the actor; otherwise Phase A
    (critic warmup) trains from a randomly-initialized actor that is only
    ever used as ``old_policy``'s reference distribution once Phase B
    begins.
    """

    # None uses the older --actor-use-layer-norm flag.
    use_layer_norm: Optional[bool] = None

    # None uses the older --actor-use-group-norm flag.
    use_group_norm: Optional[bool] = None

    # None uses the older --actor-dropout-rate flag.
    dropout_rate: Optional[float] = None


def _bppo_kwargs(
    args: Any, env_spec: OfflineEnvSpec, logger: Logger, eval_env: Any = None
) -> dict:
    from rl_garden.algorithms import BPPO

    return {
        **params_kwargs(args, BPPO.params_class),
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
        # constructor -- matches bc.py/iql.py's convention.
        "checkpoint_dir": None,
        "checkpoint_freq": 0,
        "save_replay_buffer": args.save_replay_buffer,
        "save_final_checkpoint": False,
    }


def build_bppo(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import BPPO
    from rl_garden.training.inspection import construct_agent

    agent = construct_agent(BPPO, **_bppo_kwargs(args, env_spec, logger, eval_env))
    if args.bc_checkpoint is not None:
        agent.load_actor_from(args.bc_checkpoint)
    return agent


def run_bppo(args: BPPOArgs) -> None:
    from rl_garden.envs.backend_registry import should_create_eval_env
    from rl_garden.training.offline.runner import run_offline

    if args.num_offline_steps <= args.critic_warmup_steps:
        raise SystemExit(
            "--num_offline_steps "
            f"({args.num_offline_steps}) must be greater than "
            f"--critic_warmup_steps ({args.critic_warmup_steps}), otherwise "
            "the actor-improvement phase never runs and BPPO silently "
            "degenerates into just the critic-warmup (BC-quality) policy."
        )
    if args.eval_freq <= 0 or args.env_id is None or not should_create_eval_env(args):
        raise SystemExit(
            "BPPO requires periodic real-env evaluation to gate old_policy "
            "sync (Phase B's 'safe' iteration scheme) -- pass --eval_freq "
            "> 0 and a valid --env_id/env backend so an eval env is built. "
            "Without it, old_policy never advances past its initial actor "
            "and Phase B degenerates into a single ungated PPO round."
        )

    run_offline(args, build_agent=build_bppo)




def _bppo_algorithm_cls() -> type:
    from rl_garden.algorithms import BPPO

    return BPPO

registry.register(
    "bppo",
    BPPOArgs,
    run_bppo,
    algorithm_cls=_bppo_algorithm_cls,
    build_agent=build_bppo,
    training_only_inputs=("bc_checkpoint",),
)
