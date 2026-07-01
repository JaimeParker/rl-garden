"""DrQ-v2 run function."""
from __future__ import annotations


def _drqv2_env_request(args, run_name):
    from rl_garden.envs.backend_registry import EnvRequest

    backend_config = args.resolve_backend_config()
    eval_record_dir = (
        None
        if args.eval_freq <= 0
        else args.eval_output_dir
        or f"{args.log_dir}/{run_name}/eval_videos"
    )
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        include_state=args.include_state,
        per_camera_rgbd=args.per_camera_rgbd,
        frame_stack=args.frame_stack,
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        create_eval_env=args.eval_freq > 0,
        backend_config=backend_config,
    )


def build_drqv2(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.ddpg import DDPG
    from rl_garden.encoders import discover_image_keys

    image_keys = discover_image_keys(env.single_observation_space)
    agent = DDPG(
        env=env,
        eval_env=eval_env,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        mmap_dir=args.mmap_dir,
        mmap_mode=args.mmap_mode,
        replay_lazy_next_obs=args.replay_lazy_next_obs,
        replay_pin_sampled_batch=args.replay_pin_sampled_batch,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        training_freq=args.training_freq,
        utd=args.utd,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        nstep=args.nstep,
        stddev_schedule=args.stddev_schedule,
        actor_stddev_schedule=args.actor_stddev_schedule,
        stddev_clip=args.stddev_clip,
        num_expl_steps=args.num_expl_steps,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        grad_clip_norm=args.grad_clip_norm,
        image_keys=image_keys,
        image_fusion_mode=args.image_fusion_mode,
        image_augmentation=args.image_augmentation,
        random_shift_pad=args.image_random_shift_pad,
        enable_stacking=args.frame_stack > 1,
        image_augmentation_seed=args.seed + 1_000_003,
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


def run_drqv2(args: DrQv2Args) -> None:
    from rl_garden.training.online._runner import run_online

    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    run_online(
        args,
        make_env_request=_drqv2_env_request,
        build_agent=build_drqv2,
        post_learn=lambda agent: agent.replay_buffer.flush(),
    )


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import DrQv2TrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class DrQv2Args(DrQv2TrainingArgs, EnvBackendArgs):
    """DrQ-v2 with multi-env backend support.

    ManiSkill-specific: ``--maniskill.sim_backend``, ``--maniskill.render_backend``,
    ``--maniskill.reward_mode``, ``--maniskill.success_reward_override``.
    """


registry.register("drqv2", DrQv2Args, run_drqv2)
