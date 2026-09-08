from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rl_garden.training.offline._args import (
    OfflineActorArgs,
    OfflineBPPOArgs,
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
)
from rl_garden.training.offline._registry import registry

if TYPE_CHECKING:
    from rl_garden.algorithms import OfflineEnvSpec
    from rl_garden.common import Logger


@dataclass
class BPPOArgs(
    OfflineCommonArgs,
    OfflineDeviceArgs,
    OfflineDiscountArgs,
    OfflineActorArgs,
    OfflineBPPOArgs,
):
    """BPPO (Behavior Proximal Policy Optimization) offline pretraining.

    Box observations only. Pair with ``--bc_checkpoint`` (a ``bc``
    pretrained checkpoint) to warm-start the actor; otherwise Phase A
    (critic warmup) trains from a randomly-initialized actor that is only
    ever used as ``old_policy``'s reference distribution once Phase B
    begins.
    """


def _bppo_kwargs(
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
    from rl_garden.training.offline._runner import run_offline

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


registry.register("bppo", BPPOArgs, run_bppo)
