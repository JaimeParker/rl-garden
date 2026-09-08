"""FlowPPO run function: online PPO fine-tuning for flow-matching policies.

State-only (Box observations); action chunking (``horizon_length``) is
applied here, at env construction time, via ``ActionChunkWrapper`` -- same
convention as ``rl_garden/training/online/dppo.py``, see that module's
docstring.
"""

from __future__ import annotations


def _flow_ppo_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest, should_create_eval_env

    eval_record_dir = resolve_eval_record_dir(args, run_name)
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=None,
        camera_height=None,
        include_state=True,
        per_camera_rgbd=False,
        frame_stack=1,
        num_eval_envs=args.num_eval_envs,
        create_eval_env=should_create_eval_env(args),
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=args.resolve_backend_config(),
    )


def build_flow_ppo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms import FlowPPO
    from rl_garden.envs.wrappers import ActionChunkWrapper
    from rl_garden.training.inspection import construct_agent

    env = ActionChunkWrapper(env, act_steps=args.horizon_length)
    if eval_env is not None:
        eval_env = ActionChunkWrapper(eval_env, act_steps=args.horizon_length)

    agent = construct_agent(
        FlowPPO,
        env=env,
        eval_env=eval_env,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        horizon_length=args.horizon_length,
        flow_steps=args.flow_steps,
        actor_activation_fn=args.actor_activation_fn,
        critic_activation_fn=args.critic_activation_fn,
        kernel_init=args.kernel_init,
        sde_type=args.sde_type,
        noise_level=args.noise_level,
        clip_std_min=args.clip_std_min,
        sigma_safe_max=args.sigma_safe_max,
        logprob_mode=args.logprob_mode,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        critic_warmup_updates=args.critic_warmup_updates,
        update_epochs=args.update_epochs,
        update_batch_size=args.update_batch_size,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss_coef=args.clip_vloss_coef,
        clip_advantage_lower_quantile=args.clip_advantage_lower_quantile,
        clip_advantage_upper_quantile=args.clip_advantage_upper_quantile,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        finite_horizon_gae=args.finite_horizon_gae,
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
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=_flow_ppo_env_request,
        build_agent=build_flow_ppo,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import FlowPPOTrainingArgs
from rl_garden.training.online._registry import registry


@dataclass
class FlowPPOArgs(FlowPPOTrainingArgs, EnvBackendArgs):
    """FlowPPO: online PPO fine-tuning for flow-matching policies. Trains
    ``actor``/``critic`` from scratch (no BC-checkpoint warm-start). State-
    only; ``--obs_mode`` is not exposed."""


registry.register("flow_ppo", FlowPPOArgs, run_flow_ppo)
