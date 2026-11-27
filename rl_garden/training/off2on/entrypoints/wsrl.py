"""WSRL offline-to-online training registration."""

from dataclasses import dataclass
from typing import Literal

from rl_garden.common.cli_args import PolicyModuleArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.hyperparameters.cql import WSRLParams
from rl_garden.training.off2on.args import (
    VisionWSRLTrainingArgs,
    initial_training_phase_from_args,
)
from rl_garden.training.off2on.registry import registry


@dataclass
class WSRLOff2OnArgs(VisionWSRLTrainingArgs, PolicyModuleArgs, EnvBackendArgs):
    """WSRL args. State-only observations by default; pass
    ``--obs.rgb <camera>`` for Dict/RGBD observations."""

    # Legacy widths apply only when net_arch is None.
    hidden_dim: int = 256
    actor_hidden_layers: int = 3
    critic_hidden_layers: int = 3
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"


def build_wsrl(args: WSRLOff2OnArgs, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import WSRL
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

    learning_kwargs = params_kwargs(args, WSRLParams)
    learning_kwargs["net_arch"] = net_arch

    agent = construct_agent(
        WSRL,
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
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        bootstrap_at_done=args.bootstrap_at_done,
        initial_training_phase=initial_training_phase_from_args(args),
        offline_sampling=args.offline_sampling,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
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


def run_wsrl(args: WSRLOff2OnArgs) -> None:
    from rl_garden.training.off2on.runner import run_off2on

    run_off2on(args, build_agent=build_wsrl, algorithm="wsrl")




def _wsrl_algorithm_cls() -> type:
    from rl_garden.algorithms import WSRL

    return WSRL

registry.register(
    "wsrl", WSRLOff2OnArgs, run_wsrl, algorithm_cls=_wsrl_algorithm_cls, build_agent=build_wsrl
)
