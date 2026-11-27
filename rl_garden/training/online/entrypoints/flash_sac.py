"""FlashSAC run function.

``FlashSAC`` is state-only by design (see its module docstring and
``ObservationContractError`` guard) -- ``FlashSACPolicy`` has no encoder
mixin/Dict handling. ``FlashSACArgs`` still carries ``ObservationArgs`` for
CLI/config uniformity (``--obs.state`` parses), but ``build_flash_sac`` never
wires an encoder; ``--obs.rgb``/``--obs.depth`` reach the env, then fail fast
at agent construction rather than being silently ignored.
"""

from __future__ import annotations


def build_flash_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.hyperparameters import params_kwargs
    from rl_garden.hyperparameters.flash_sac import FlashSACParams
    from rl_garden.algorithms.flash_sac import FlashSAC
    from rl_garden.training.inspection import construct_agent

    agent = construct_agent(
        FlashSAC,
        **params_kwargs(args, FlashSACParams),
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        training_freq=args.training_freq,
        utd=args.utd,
        asymmetric_obs_dim=args.asymmetric_obs_dim,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        use_amp=args.use_amp,
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
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent


def run_flash_sac(args: FlashSACArgs) -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=make_env_request,
        build_agent=build_flash_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import FlashSACTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class FlashSACArgs(FlashSACTrainingArgs, ObservationArgs, EnvBackendArgs):
    """FlashSAC with multi-env backend support (state-only by design).

    ManiSkill-specific: ``--maniskill.sim-backend``, ``--maniskill.render-backend``.
    """


def _flash_sac_algorithm_cls() -> type:
    from rl_garden.algorithms.flash_sac import FlashSAC

    return FlashSAC


registry.register(
    "flash_sac",
    FlashSACArgs,
    run_flash_sac,
    algorithm_cls=_flash_sac_algorithm_cls,
    build_agent=build_flash_sac,
)
