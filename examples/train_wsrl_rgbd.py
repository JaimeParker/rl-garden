"""Vision-based WSRL on ManiSkill with offline→online training.

Usage:
    # RGB observations with plain_conv encoder
    python examples/train_wsrl_rgbd.py --env_id PickCube-v1 --obs_mode rgb --encoder plain_conv

    # RGBD observations with ResNet encoder
    python examples/train_wsrl_rgbd.py --env_id PickCube-v1 --obs_mode rgbd --encoder resnet10

    # Online-only training (no offline pre-training)
    python examples/train_wsrl_rgbd.py --env_id PickCube-v1 --num_offline_steps 0

    # Offline→online training from a ManiSkill trajectory H5
    python examples/train_wsrl_rgbd.py --env_id PickCube-v1 --offline_dataset_path demos/pickcube_rgb.h5 --num_offline_steps 100000
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import tyro
import os
from tqdm import trange

from rl_garden.algorithms import WSRLRGBD
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionWSRLTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(VisionWSRLTrainingArgs):
    pass


def _offline_update_loop(
    agent: WSRLRGBD, steps: int, logger: Logger, log_freq: int, std_log: bool
) -> None:
    gradient_steps = int(agent.utd) if float(agent.utd).is_integer() and agent.utd > 1 else 1
    for step in trange(steps, desc="offline"):
        losses = agent.train(gradient_steps)
        if log_freq > 0 and (step + 1) % log_freq == 0:
            for key, value in losses.items():
                logger.add_scalar(f"offline_losses/{key}", value, step + 1)
            if std_log:
                progress = 100.0 * (step + 1) / steps if steps > 0 else 100.0
                loss_summary = " ".join(
                    f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float))
                )
                print(
                    "[offline] "
                    f"step={step + 1}/{steps} ({progress:.2f}%) {loss_summary}",
                    flush=True,
                )


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__wsrl_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
    )
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
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id, num_envs=args.num_envs,
        obs_mode=args.obs_mode, include_state=args.include_state,
        control_mode=args.control_mode,
        camera_width=args.camera_width, camera_height=args.camera_height,
        render_mode=args.render_mode,
    )
    eval_record_dir = args.eval_output_dir or os.path.join(args.log_dir, run_name, "eval_videos")
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id, num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode, include_state=args.include_state,
        control_mode=args.control_mode, reconfiguration_freq=1,
        camera_width=args.camera_width, camera_height=args.camera_height,
        render_mode=args.render_mode,
        record_dir=eval_record_dir,
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)

    agent = WSRLRGBD(
        env=env, eval_env=eval_env,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, tau=args.tau,
        training_freq=args.training_freq, utd=args.utd,
        policy_lr=args.policy_lr, q_lr=args.q_lr,
        alpha_lr=args.alpha_lr, cql_alpha_lr=args.cql_alpha_lr,
        n_critics=args.n_critics,
        critic_subsample_size=args.critic_subsample_size,
        use_cql_loss=args.use_cql_loss,
        cql_n_actions=args.cql_n_actions,
        cql_alpha=args.cql_alpha,
        cql_autotune_alpha=args.cql_autotune_alpha,
        cql_alpha_lagrange_init=args.cql_alpha_lagrange_init,
        cql_target_action_gap=args.cql_target_action_gap,
        cql_importance_sample=args.cql_importance_sample,
        cql_max_target_backup=args.cql_max_target_backup,
        cql_temp=args.cql_temp,
        cql_clip_diff_min=args.cql_clip_diff_min,
        cql_clip_diff_max=args.cql_clip_diff_max,
        cql_action_sample_method=args.cql_action_sample_method,
        backup_entropy=args.backup_entropy,
        use_calql=args.use_calql,
        calql_bound_random_actions=args.calql_bound_random_actions,
        actor_use_layer_norm=args.actor_use_layer_norm,
        critic_use_layer_norm=args.critic_use_layer_norm,
        std_parameterization=args.std_parameterization,
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        seed=args.seed, logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )

    # Offline training phase
    if args.num_offline_steps > 0:
        if args.offline_dataset_path is None:
            raise ValueError("--offline_dataset_path is required when --num_offline_steps > 0.")
        loaded = load_maniskill_h5_to_replay_buffer(
            agent.replay_buffer,
            args.offline_dataset_path,
            num_traj=args.offline_num_traj,
        )
        logger.add_scalar("offline/loaded_transitions", loaded, 0)
        _offline_update_loop(
            agent, args.num_offline_steps, logger, args.log_freq, args.std_log
        )

        # Switch to online mode
        agent.switch_to_online_mode()

    # Online training phase
    if args.num_online_steps > 0:
        agent.learn(total_timesteps=args.num_online_steps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
