"""Standard FQL offline-to-online registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters import params_kwargs
from rl_garden.hyperparameters.fql import FQLReferenceParams
from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.off2on.args import FQLTrainingArgs
from rl_garden.training.off2on.registry import registry


@dataclass
class FQLOff2OnArgs(FQLReferenceParams, FQLTrainingArgs, ObservationArgs, EnvBackendArgs):
    """Standard FQL continues offline-to-online without interruption by default.

    Positive ``warmup_steps`` pauses updates during initial online collection."""

    hidden_dim: int = 512
    hidden_layers: int = 4
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    num_eval_episodes: Optional[int] = None
    learning_starts: int = 0
    training_freq: int = 1
    utd: float = 1.0
    online_replay_mode: Literal["empty", "append", "mixed"] = "append"
    warmup_steps: int = 0


def build_fql(args, env, eval_env, logger, checkpoint_dir):
    import torch

    from rl_garden.algorithms import Off2OnFQL
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.off2on.args import initial_training_phase_from_args

    if args.reference_compatible:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing
    learning_kwargs = params_kwargs(args, Off2OnFQL.params_class)
    learning_kwargs["net_arch"] = list(args.net_arch) if args.net_arch is not None else [args.hidden_dim] * args.hidden_layers

    agent = construct_agent(
        Off2OnFQL,
        **learning_kwargs,
        env=env,
        eval_env=eval_env,
        **image_kwargs,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
        bootstrap_at_done=args.bootstrap_at_done,
        actor_use_group_norm=args.actor_use_group_norm,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.online_eval_freq or 0,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_fql(args: FQLOff2OnArgs) -> None:
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_fql, algorithm="fql")


def _off2on_fql_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnFQL

    return Off2OnFQL


registry.register(
    "fql", FQLOff2OnArgs, run_fql, algorithm_cls=_off2on_fql_algorithm_cls, build_agent=build_fql
)
