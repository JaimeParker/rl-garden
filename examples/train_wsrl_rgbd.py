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
from tqdm import trange

from rl_garden.algorithms import WSRLRGBD
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionWSRLTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(VisionWSRLTrainingArgs):
    pass


_Q_METRIC_LABELS = {
    "predicted_q": "predicted",
    "target_q": "target",
    "cql_ood_values": "cql_ood",
    "cql_q_diff": "cql_diff",
}


def _split_q_metrics(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    q_metrics: dict[str, float] = {}
    other_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if key in _Q_METRIC_LABELS:
            q_metrics[_Q_METRIC_LABELS[key]] = float(value)
        else:
            other_metrics[key] = float(value)
    return other_metrics, q_metrics


def _metric_summary(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def _log_offline_metrics(
    logger: Logger,
    metrics: dict[str, float],
    step: int,
) -> None:
    other_metrics, q_metrics = _split_q_metrics(metrics)
    for key, value in other_metrics.items():
        logger.add_scalar(f"losses/{key}", value, step)
    for key, value in q_metrics.items():
        logger.add_scalar(f"q/{key}", value, step)


def _offline_update_loop(
    agent: WSRLRGBD, steps: int, logger: Logger, log_freq: int, std_log: bool
) -> None:
    gradient_steps = (
        int(agent.utd) if float(agent.utd).is_integer() and agent.utd > 1 else 1
    )
    interval_update_time = 0.0
    interval_update_steps = 0
    for step in trange(steps, desc="offline"):
        update_t = time.perf_counter()
        losses = agent.train(gradient_steps)
        interval_update_time += time.perf_counter() - update_t
        interval_update_steps += gradient_steps
        if log_freq > 0 and (step + 1) % log_freq == 0:
            _log_offline_metrics(logger, losses, step + 1)
            offline_update_fps = (
                interval_update_steps / interval_update_time
                if interval_update_time > 0
                else float("nan")
            )
            logger.add_scalar("time/offline_update_time", interval_update_time, step + 1)
            logger.add_scalar("time/offline_update_fps", offline_update_fps, step + 1)
            if std_log:
                progress = 100.0 * (step + 1) / steps if steps > 0 else 100.0
                other_metrics, q_metrics = _split_q_metrics(losses)
                loss_summary = _metric_summary(other_metrics)
                q_summary = _metric_summary(q_metrics)
                q_part = f" q={q_summary}" if q_summary else ""
                print(
                    "[offline] "
                    f"step={step + 1}/{steps} ({progress:.2f}%) "
                    f"fps={offline_update_fps:.4f} "
                    f"{loss_summary}{q_part}",
                    flush=True,
                )
            interval_update_time = 0.0
            interval_update_steps = 0


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__wsrl_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
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
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
    )
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
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)

    agent = WSRLRGBD(
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
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        cql_alpha_lr=args.cql_alpha_lr,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip_norm=args.grad_clip_norm,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
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
        actor_use_group_norm=args.actor_use_group_norm,
        critic_use_group_norm=args.critic_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        critic_dropout_rate=args.critic_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        std_parameterization=args.std_parameterization,
        online_cql_alpha=args.online_cql_alpha,
        online_use_cql_loss=args.online_use_cql_loss,
        offline_sampling=args.offline_sampling,
        sparse_reward_mc=args.sparse_reward_mc,
        sparse_negative_reward=args.sparse_negative_reward,
        success_threshold=args.success_threshold,
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
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)

    # Offline training phase
    if args.num_offline_steps > 0:
        if args.offline_dataset_path is None:
            raise ValueError(
                "--offline_dataset_path is required when --num_offline_steps > 0."
            )
        loaded = load_maniskill_h5_to_replay_buffer(
            agent.replay_buffer,
            args.offline_dataset_path,
            num_traj=args.offline_num_traj,
            reward_scale=args.reward_scale,
            reward_bias=args.reward_bias,
            success_key=args.success_key,
        )
        logger.add_scalar("offline/loaded_transitions", loaded, 0)
        _offline_update_loop(
            agent, args.num_offline_steps, logger, args.log_freq, args.std_log
        )

        # Switch to online mode
        agent.switch_to_online_mode(
            online_replay_mode=args.online_replay_mode,
            offline_data_ratio=args.offline_data_ratio,
        )

    # Online training phase
    if args.num_online_steps > 0:
        online_target_step = agent._global_step + args.num_online_steps
        agent.learn(total_timesteps=online_target_step)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
