"""State-based FlashSAC on ManiSkill.

Usage (from repo root):
    python examples/train_flash_sac_state.py --env_id PickCube-v1 --num_envs 512
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import tyro

from rl_garden.algorithms.flash_sac import FlashSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import LoggingArgs, apply_log_env_overrides
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class FlashSACArgs:
    # --- environment ---
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 8
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"

    # --- replay / training ---
    buffer_size: int = 10_000_000
    buffer_device: str = "cuda"
    learning_starts: int = 4_000
    batch_size: int = 2048
    gamma: float = 0.99
    tau: float = 0.005
    training_freq: int = 512
    utd: float = 1.0
    n_step: int = 3
    total_timesteps: int = 10_000_000

    # --- architecture ---
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    num_bins: int = 101
    min_v: float = -5.0
    max_v: float = 5.0
    asymmetric_obs_dim: int = 0

    # --- optimizers ---
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    actor_update_period: int = 1
    grad_clip_norm: Optional[float] = None

    # --- temperature ---
    temp_initial_value: float = 0.01
    target_entropy: str = "auto"

    # --- exploration ---
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16

    # --- reward normalization ---
    normalize_reward: bool = False
    normalized_g_max: float = 10.0

    # --- BC regularization ---
    bc_alpha: float = 0.0

    # --- performance ---
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False

    # --- eval / video ---
    capture_video: bool = False
    video_fps: int = 20

    # --- seed ---
    seed: int = 1

    # --- logging (nested, controls log_dir / wandb / tensorboard / etc.) ---
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # --- checkpoint ---
    checkpoint_freq: int = 0
    save_replay_buffer: bool = False
    save_final_checkpoint: bool = True
    load_checkpoint: Optional[str] = None


def main() -> None:
    args = tyro.cli(FlashSACArgs)
    apply_log_env_overrides(args.logging)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.logging.exp_name
        or f"{args.env_id}__flash_sac_state__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = (
        f"{args.logging.log_dir}/{run_name}/checkpoints"
        if args.save_final_checkpoint or args.checkpoint_freq > 0
        else None
    )
    logger = Logger.create(
        log_type=args.logging.log_type,
        log_dir=args.logging.log_dir,
        run_name=run_name,
        config=vars(args),
        start_time=start_time,
        log_keywords=args.logging.log_keywords,
        wandb_project=args.logging.wandb_project,
        wandb_entity=args.logging.wandb_entity,
        wandb_group=args.logging.wandb_group or args.env_id,
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=args.render_mode,
    )
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        render_mode=args.render_mode,
        reconfiguration_freq=1,
        record_dir=f"{args.logging.log_dir}/{run_name}/videos" if args.capture_video else None,
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.logging.num_eval_steps,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

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
        std_log=args.logging.std_log,
        log_freq=args.logging.log_freq,
        eval_freq=args.logging.eval_freq,
        num_eval_steps=args.logging.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )

    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint)

    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
