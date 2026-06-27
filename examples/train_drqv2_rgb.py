"""DrQ-v2 (DDPG + augmentation + n-step) RGB training entrypoint.

Usage:
    python examples/train_drqv2_rgb.py --env_id PickCube-v1
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro

from rl_garden.algorithms.ddpg import DDPG
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.encoders import discover_image_keys
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
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
    sim_backend: str = "gpu"
    render_backend: str = "gpu"

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
    enable_stacking: bool = False

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


def main() -> None:
    args = tyro.cli(Args)
    if args.mmap_dir is not None and args.load_replay_buffer:
        raise SystemExit(
            "--load-replay-buffer is not supported with --mmap-dir; "
            "use --mmap-mode open to resume the disk-backed buffer"
        )
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__drqv2__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=vars(args),
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group or args.env_id,
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        render_mode=args.render_mode,
        per_camera_rgbd=args.per_camera_rgbd,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    env = make_maniskill_env(env_cfg)
    image_keys = discover_image_keys(env.single_observation_space)
    eval_env = None
    if args.eval_freq > 0:
        eval_record_dir = resolve_eval_record_dir(args, run_name)
        eval_cfg = ManiSkillEnvConfig(
            env_id=args.env_id,
            num_envs=args.num_eval_envs,
            obs_mode=args.obs_mode,
            include_state=args.include_state,
            control_mode=args.control_mode,
            reconfiguration_freq=1,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            render_mode=args.render_mode,
            record_dir=eval_record_dir,
            save_video=args.capture_video,
            video_fps=args.video_fps,
            max_steps_per_video=args.num_eval_steps,
            per_camera_rgbd=args.per_camera_rgbd,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
        eval_env = make_maniskill_env(eval_cfg)

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
        enable_stacking=args.enable_stacking,
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
    agent.learn(total_timesteps=args.total_timesteps)
    agent.replay_buffer.flush()

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
