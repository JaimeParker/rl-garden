"""FlashSAC run function."""
from __future__ import annotations


def _flash_sac_env_request(args, run_name):
    from rl_garden.envs.backend_registry import EnvRequest

    backend_config = args.resolve_backend_config()
    eval_record_dir = f"{args.log_dir}/{run_name}/videos" if args.capture_video else None
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
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )


def build_flash_sac(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.flash_sac import FlashSAC

    agent = FlashSAC(
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        n_step=args.n_step,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_blocks=args.actor_num_blocks,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_blocks=args.critic_num_blocks,
        num_bins=args.num_bins,
        min_v=args.min_v,
        max_v=args.max_v,
        asymmetric_obs_dim=args.asymmetric_obs_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        actor_update_period=args.actor_update_period,
        grad_clip_norm=args.grad_clip_norm,
        temp_initial_value=args.temp_initial_value,
        target_entropy=args.target_entropy,
        actor_noise_zeta_mu=args.actor_noise_zeta_mu,
        actor_noise_zeta_max=args.actor_noise_zeta_max,
        normalize_reward=args.normalize_reward,
        normalized_g_max=args.normalized_g_max,
        bc_alpha=args.bc_alpha,
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
        agent.load(args.load_checkpoint)
    return agent


def run_flash_sac(args: FlashSACArgs) -> None:
    from rl_garden.training.online._runner import run_online

    run_online(
        args,
        obs_tag="state",
        make_env_request=_flash_sac_env_request,
        build_agent=build_flash_sac,
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import FlashSACTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class FlashSACArgs(FlashSACTrainingArgs, EnvBackendArgs):
    """FlashSAC with multi-env backend support (state-only).

    ManiSkill-specific: ``--maniskill.sim_backend``, ``--maniskill.render_backend``.
    """


registry.register("flash_sac", FlashSACArgs, run_flash_sac)
