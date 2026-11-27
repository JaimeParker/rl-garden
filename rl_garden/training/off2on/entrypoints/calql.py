"""Cal-QL offline-to-online training registration.

Builds ``Off2OnCalQL`` (a thin subclass of the shared off2on Cal-QL shell that
also backs ``WSRL``) instead of ``WSRL`` itself, and reuses the same
``_runner.run_off2on`` orchestration. This entrypoint's default preset has
no warmup, retains offline data mixed throughout online fine-tuning, and
keeps the CQL/Cal-QL regularizer online — matching Nakamoto et al. 2023
(Cal-QL) instead of the WSRL paper's warmup-then-discard preset used by the
`wsrl` entrypoint.
"""

from dataclasses import dataclass
from typing import Literal

from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.hyperparameters.cql import Off2OnCalQLParams
from rl_garden.training.off2on.args import (
    VisionWSRLTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on.registry import registry


@dataclass
class CalQLOff2OnArgs(Off2OnCalQLParams, VisionWSRLTrainingArgs, PolicyModuleArgs, EnvBackendArgs):
    """Cal-QL off2on args: no warmup, mixed replay, CQL retained online.

    State-only observations by default; pass ``--obs.rgb <camera>`` for
    Dict/RGBD observations.
    """

    warmup_steps: int = 0
    online_replay_mode: Literal["empty", "append", "mixed"] = "mixed"
    offline_data_ratio: float | str = "auto"
    # Cal-QL uses two hidden layers when net_arch is None.
    hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 2

    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    online_episodes_per_iteration: int | None = None
    stats_window_size: int | None = None
    num_eval_episodes: int | None = None


def build_calql(args: CalQLOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import Off2OnCalQL
    from rl_garden.common.cli_args import (
        policy_module_kwargs,
        resolve_critic_encoder_config,
        resolve_obs_groups_config,
    )
    from rl_garden.training.inspection import construct_agent

    image_kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
    }
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    from rl_garden.hyperparameters import params_kwargs

    net_arch = args.net_arch if args.net_arch is not None else {
        "pi": [args.hidden_dim] * args.actor_hidden_layers,
        "qf": [args.hidden_dim] * args.critic_hidden_layers,
    }

    learning_kwargs = params_kwargs(args, Off2OnCalQLParams)
    learning_kwargs["net_arch"] = net_arch

    agent = construct_agent(
        Off2OnCalQL,
        **learning_kwargs,
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
        bootstrap_at_done=args.bootstrap_at_done,
        online_episodes_per_iteration=args.online_episodes_per_iteration,
        stats_window_size=args.stats_window_size,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
        **image_kwargs,
        **policy_module_kwargs(args),
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_calql(args: CalQLOff2OnArgs) -> None:
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_calql, algorithm="calql")


def _off2_on_calql_algorithm_cls() -> type:
    from rl_garden.algorithms import Off2OnCalQL

    return Off2OnCalQL


registry.register(
    "calql",
    CalQLOff2OnArgs,
    run_calql,
    algorithm_cls=_off2_on_calql_algorithm_cls,
    build_agent=build_calql,
)
