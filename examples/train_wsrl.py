"""State-based WSRL on ManiSkill with offline→online training.

Usage:
    # Online-only (no offline pre-training)
    python examples/train_wsrl.py --env_id PickCube-v1 --num_offline_steps 0

    # Offline→online training
    python examples/train_wsrl.py --env_id PickCube-v1 --offline_dataset_path demos/pickcube.h5 --num_offline_steps 100000 --num_online_steps 50000

    # Disable REDQ (use 2 critics like standard SAC)
    python examples/train_wsrl.py --env_id PickCube-v1 --n_critics 2
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import trange

from rl_garden.algorithms import WSRL
from rl_garden.buffers import MCDictReplayBuffer, MCTensorReplayBuffer
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    WSRLTrainingArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(WSRLTrainingArgs):
    pass


def _offline_update_loop(
    agent: WSRL,
    steps: int,
    logger: Logger,
    log_freq: int,
    std_log: bool,
    *,
    start_step: int = 0,
) -> int:
    gradient_steps = (
        int(agent.utd) if float(agent.utd).is_integer() and agent.utd > 1 else 1
    )
    for step in trange(steps, desc="offline"):
        global_step = start_step + step + 1
        losses = agent.train(gradient_steps)
        if log_freq > 0 and (step + 1) % log_freq == 0:
            agent._log_update_metrics(losses, global_step)
            if std_log:
                progress = 100.0 * (step + 1) / steps if steps > 0 else 100.0
                loss_summary = " ".join(
                    f"{k}={v:.4f}"
                    for k, v in losses.items()
                    if isinstance(v, (int, float))
                )
                print(
                    "[offline] "
                    f"step={step + 1}/{steps} "
                    f"global_step={global_step} "
                    f"({progress:.2f}%) {loss_summary}",
                    flush=True,
                )
    return start_step + steps


def _first_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _evaluate_offline_end(agent: WSRL, logger: Logger, step: int, std_log: bool) -> None:
    metrics = agent._evaluate()
    if not metrics:
        return
    agent._log_eval_metrics(metrics, step)
    for key, value in agent.canonical_eval_metrics(metrics).items():
        logger.add_summary(f"wsrl/offline_final_eval/{key}", value)
    if std_log:
        eval_return = _first_metric(metrics, ("return",))
        eval_success = _first_metric(metrics, ("success_at_end", "success_once"))
        print(
            "[offline_eval] "
            f"step={step} "
            f"return={_fmt_metric(eval_return)} "
            f"success_at_end={_fmt_metric(eval_success)}",
            flush=True,
        )


def _save_offline_checkpoint(
    agent: WSRL,
    checkpoint_dir: str | None,
    *,
    include_replay_buffer: bool,
    std_log: bool,
) -> None:
    if checkpoint_dir is None:
        return
    path = agent.save(
        Path(checkpoint_dir) / "offline_final.pt",
        include_replay_buffer=include_replay_buffer,
    )
    if std_log:
        print(f"[offline] saved_checkpoint={path}", flush=True)


def _clear_replay_buffer(agent: WSRL, logger: Logger, step: int, std_log: bool) -> int:
    """Start online replay from an empty buffer, matching upstream WSRL default."""
    buffer = agent.replay_buffer
    previous_len = len(buffer)
    buffer.pos = 0
    buffer.full = False
    if isinstance(buffer, (MCTensorReplayBuffer, MCDictReplayBuffer)):
        buffer._mc_table = None
    logger.add_summary("wsrl/online_replay_mode", "empty")
    logger.add_summary("wsrl/online_replay_cleared", True)
    logger.add_summary("wsrl/online_replay_size_before_clear", previous_len)
    if std_log:
        print(f"[online] replay_mode=empty cleared_transitions={previous_len}", flush=True)
    return previous_len


def _set_offline_probe(agent: WSRL, logger: Logger, std_log: bool) -> None:
    probe_size = min(agent.batch_size, len(agent.replay_buffer))
    if probe_size <= 0:
        logger.add_summary("wsrl/offline_probe_size", 0)
        return
    agent.set_offline_probe_batch(agent.replay_buffer.sample(probe_size))
    logger.add_summary("wsrl/offline_probe_size", probe_size)
    if std_log:
        print(f"[offline] probe_size={probe_size}", flush=True)


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name or f"{args.env_id}__wsrl_state__{args.seed}__{int(time.time())}"
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
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        reward_scale=args.reward_scale,
        reward_bias=args.reward_bias,
    )
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        reconfiguration_freq=1,
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

    agent = WSRL(
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
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
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
        )
        offline_start_step = agent._global_step
        logger.add_summary("wsrl/offline_loaded_transitions", loaded)
        logger.add_summary("wsrl/offline_start_step", offline_start_step)
        offline_end_step = _offline_update_loop(
            agent,
            args.num_offline_steps,
            logger,
            args.log_freq,
            args.std_log,
            start_step=offline_start_step,
        )
        agent._global_step = offline_end_step
        _evaluate_offline_end(agent, logger, offline_end_step, args.std_log)
        _save_offline_checkpoint(
            agent,
            checkpoint_dir,
            include_replay_buffer=args.save_replay_buffer,
            std_log=args.std_log,
        )
        _set_offline_probe(agent, logger, args.std_log)
        if args.online_replay_mode == "empty":
            _clear_replay_buffer(agent, logger, offline_end_step, args.std_log)
        elif args.online_replay_mode == "append":
            logger.add_summary("wsrl/online_replay_mode", "append")
            logger.add_summary("wsrl/online_replay_cleared", False)

        # Switch to online mode (passes mode + ratio so mixed-batch is wired up).
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
