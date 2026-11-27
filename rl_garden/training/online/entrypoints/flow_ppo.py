"""FlowPPO run function: online PPO fine-tuning for flow-matching policies.

State-only observations by default; pass ``--obs.rgb <camera>`` for
CNN-based Dict/RGBD observations. Action chunking (``horizon_length``) is
applied here, at env construction time, via ``ActionChunkWrapper`` -- same
convention as ``rl_garden/training/online/entrypoints/dppo.py``, see that module's
docstring.
"""

from __future__ import annotations

from rl_garden.hyperparameters import params_kwargs


def build_flow_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.common.cli_args import resolve_critic_encoder_config, resolve_obs_groups_config
    from rl_garden.algorithms import FlowPPO
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.horizon_length)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.horizon_length)

    image_kwargs = dict(
        encoder_config=args.encoder if args.obs.is_visual else None,
        obs_groups=resolve_obs_groups_config(args),
        critic_encoder_config=resolve_critic_encoder_config(args),
    )
    if args.encoder_sharing is not None:
        image_kwargs["encoder_sharing"] = args.encoder_sharing

    agent = construct_agent(
        FlowPPO,
        **params_kwargs(args, FlowPPO.params_class),
        env=env,
        eval_env=eval_env,
        **image_kwargs,
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


def run_flow_ppo(args: "FlowPPOArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.runner import run_online

    obs_tag = f"rgbd_{args.encoder.backbone}" if args.obs.is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=make_env_request,
        build_agent=build_flow_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.cli_args import ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online.args import FlowPPOTrainingArgs
from rl_garden.training.online.registry import registry


@dataclass
class FlowPPOArgs(FlowPPOTrainingArgs, ObservationArgs, EnvBackendArgs):
    """FlowPPO: online PPO fine-tuning for flow-matching policies. Trains
    ``actor``/``critic`` from scratch (no BC-checkpoint warm-start).
    State-only observations by default; pass ``--obs.rgb <camera>`` for
    CNN-based Dict/RGBD observations."""


def _flow_ppo_algorithm_cls() -> type:
    from rl_garden.algorithms import FlowPPO

    return FlowPPO


registry.register(
    "flow_ppo",
    FlowPPOArgs,
    run_flow_ppo,
    algorithm_cls=_flow_ppo_algorithm_cls,
    build_agent=build_flow_ppo,
)
