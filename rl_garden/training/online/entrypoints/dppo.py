"""DPPO (Diffusion PPO) fine-tuning run function.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Action chunking is applied here, at env
construction time, via ``ActionChunkWrapper`` -- ``DPPO`` itself only ever
sees an already-chunked ``env.single_action_space`` (see
``rl_garden/algorithms/dppo.py``'s module docstring).
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs


def _dppo_observation_kwargs(args) -> dict:
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config

    kwargs: dict = {
        "encoder_config": args.encoder if args.obs.is_visual else None,
        "obs_groups": resolve_obs_groups_config(args),
        "critic_encoder_config": resolve_critic_encoder_config(args),
        "image_augmentation_seed": args.seed + 1_000_003,
    }
    if args.encoder_sharing is not None:
        kwargs["encoder_sharing"] = args.encoder_sharing
    return kwargs


def build_dppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import DPPO
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.act_steps)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.act_steps)

    image_kwargs = _dppo_observation_kwargs(args)

    agent = construct_agent(
        DPPO,
        **params_kwargs(args, DPPO.params_class),
        env=env,
        eval_env=eval_env,
        **image_kwargs,
        bc_checkpoint=args.bc_checkpoint or None,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        finite_horizon_gae=args.finite_horizon_gae,
        bootstrap_at_done=args.bootstrap_at_done,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=False)
    return agent


def run_dppo(args: "DPPOArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_dppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import DPPOTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class DPPOArgs(DPPOTrainingArgs, ObservationArgs, EnvBackendArgs):
    """DPPO (Diffusion PPO) fine-tuning. Requires ``--bc_checkpoint`` (a
    ``DiffusionBC`` checkpoint -- state-only obs, so a checkpoint trained
    against Box obs will not load into a Dict-obs DPPO run). State-only
    observations by default; pass ``--obs.rgb <camera>`` for CNN-based
    Dict/RGBD observations."""


def _dppo_algorithm_cls() -> type:
    from rl_garden.algorithms import DPPO

    return DPPO


registry.register(
    "dppo",
    DPPOArgs,
    run_dppo,
    algorithm_cls=_dppo_algorithm_cls,
    build_agent=build_dppo,
    training_only_inputs=("bc_checkpoint",),
)
