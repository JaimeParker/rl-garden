"""State-based PORL fine-tuning with a pre-trained policy checkpoint.

Usage:
    python examples/train_porl.py \
        --env_id PickCube-v1 \
        --load_checkpoint runs/pretrained_policy/final.pt \
        --porl_pre_sample_steps 5000 \
        --num_online_steps 1000000
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import tyro

from rl_garden.algorithms import WSRL
from rl_garden.common import Logger, enable_fast_math, seed_everything
from rl_garden.common.cli_args import (
    WSRLTrainingArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(WSRLTrainingArgs):
    num_offline_steps: int = 0
    online_replay_mode: Literal["empty"] = "empty"
    offline_data_ratio: float = 0.0
    use_cql_loss: bool = False
    use_calql: bool = False
    online_use_cql_loss: bool = False
    warmup_steps: int = 0
    porl_pre_sample_steps: int = 5_000
    porl_epsilon: float = 0.1


def _validate_porl_args(args: Args) -> None:
    if args.load_checkpoint is None:
        raise ValueError(
            "PORL requires --load_checkpoint pointing to a pre-trained policy."
        )
    if args.num_offline_steps != 0 or args.offline_dataset_path is not None:
        raise ValueError(
            "PORL is policy-only: do not pass offline data or offline steps. "
            "Use examples/train_wsrl.py for WSRL offline-to-online training."
        )
    if args.offline_data_ratio != 0.0:
        raise ValueError("PORL uses online replay only; offline_data_ratio must be 0.")
    if args.use_cql_loss or args.use_calql or args.online_use_cql_loss:
        raise ValueError("PORL runs SAC-style online updates; CQL/CalQL must be off.")
    if args.warmup_steps != 0:
        raise ValueError("PORL pre-sample replaces WSRL warmup; warmup_steps must be 0.")


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    _validate_porl_args(args)
    seed_everything(args.seed)
    enable_fast_math()

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__porl_state__{args.seed}__{int(time.time())}"
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

    if args.eval_freq > 0 and args.num_eval_envs <= 0:
        raise SystemExit(
            "--eval_freq > 0 requires --num_eval_envs > 0 to provide an eval environment."
        )

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
        warmup_steps=args.warmup_steps,
        offline_sampling=args.offline_sampling,
        porl_pre_sample_steps=args.porl_pre_sample_steps,
        porl_epsilon=args.porl_epsilon,
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
    agent.load_actor_checkpoint(args.load_checkpoint)
    agent.switch_to_online_mode(online_replay_mode="empty", offline_data_ratio=0.0)

    if args.std_log:
        print(
            "[porl] "
            f"actor_checkpoint={args.load_checkpoint} "
            f"pre_sample_steps={args.porl_pre_sample_steps} "
            f"epsilon={args.porl_epsilon}",
            flush=True,
        )

    if args.num_online_steps > 0:
        online_target_step = agent._global_step + args.num_online_steps
        agent.learn(total_timesteps=online_target_step)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
