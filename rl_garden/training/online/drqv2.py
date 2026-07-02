"""DrQ-v2 run function."""
from __future__ import annotations

import warnings
from typing import Literal, Optional


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
    from rl_garden.common.cli_args import image_encoder_factory_from_args
    from rl_garden.encoders import discover_image_keys

    if args.encoder != "drqv2_conv":
        warnings.warn(
            f"DrQv2's validated default encoder is 'drqv2_conv'; overriding "
            f"with --encoder {args.encoder!r} deviates from the DrQ-v2 paper "
            "architecture.",
            stacklevel=2,
        )
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
        image_encoder_factory=image_encoder_factory_from_args(args),
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
class DrQv2Args(EnvBackendArgs):
    """DrQ-v2 with multi-env backend support.

    ManiSkill-specific: ``--maniskill.sim_backend``, ``--maniskill.render_backend``,
    ``--maniskill.reward_mode``.
    """

    # --- Env ---
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    obs_mode: str = "rgbd"
    include_state: bool = True
    control_mode: str = "pd_ee_delta_pose"
    camera_width: int = 128
    camera_height: int = 128
    render_mode: str = "rgb_array"
    per_camera_rgbd: bool = False

    # --- Training ---
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    mmap_dir: Optional[str] = None
    mmap_mode: Literal["create", "open"] = "create"
    learning_starts: int = 4_000
    batch_size: int = 256
    seed: int = 1

    # --- DDPG ---
    gamma: float = 0.99
    tau: float = 0.01
    training_freq: int = 32
    utd: float = 0.5
    policy_lr: float = 1e-4
    q_lr: float = 1e-4
    feature_dim: int = 50
    hidden_dim: int = 1024
    nstep: int = 3
    stddev_schedule: str = "linear(1.0,0.1,500000)"
    stddev_clip: float = 0.3
    num_expl_steps: int = 2000
    grad_clip_norm: Optional[float] = None
    weight_decay: float = 0.0
    use_adamw: bool = False

    # --- Vision ---
    image_fusion_mode: str = "stack_channels"
    image_augmentation: str = "random_shift"
    image_random_shift_pad: int = 4
    frame_stack: int = 1
    encoder: Literal["drqv2_conv", "cnn3d"] = "drqv2_conv"
    encoder_features_dim: int = 256
    # Unused for drqv2_conv/cnn3d; required attributes for
    # image_encoder_factory_from_args()'s resnet-only-flag validation.
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    # --- Logging ---
    log_type: str = "wandb"
    log_dir: str = "runs"
    exp_name: str = ""
    wandb_project: str = "rl-garden"
    wandb_entity: str = ""
    wandb_group: str = ""
    log_keywords: str = ""
    std_log: bool = True
    log_freq: int = 1_000

    # --- Eval ---
    eval_freq: int = 10_000
    num_eval_steps: int = 50
    capture_video: bool = True
    eval_output_dir: Optional[str] = None
    video_fps: int = 30

    # --- Checkpoint ---
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 0
    save_replay_buffer: bool = False
    save_final_checkpoint: bool = True
    load_checkpoint: Optional[str] = None
    load_replay_buffer: bool = False
    replay_lazy_next_obs: bool = False
    replay_pin_sampled_batch: bool = False


registry.register("drqv2", DrQv2Args, run_drqv2)
