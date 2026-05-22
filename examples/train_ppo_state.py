"""State-based PPO on ManiSkill."""

from __future__ import annotations

import time
from dataclasses import dataclass

import tyro

from rl_garden.algorithms import PPO
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    PPOTrainingArgs,
    apply_log_env_overrides,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(PPOTrainingArgs):
    pass


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name or f"{args.env_id}__ppo_state__{args.seed}__{int(time.time())}"
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
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_eval_envs,
        obs_mode="state",
        control_mode=args.control_mode,
        reconfiguration_freq=1,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        record_dir=eval_record_dir,
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    agent = PPO(
        env=env,
        eval_env=eval_env,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        norm_adv=args.norm_adv,
        clip_coef=args.clip_coef,
        clip_vloss=args.clip_vloss,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        anneal_lr=args.anneal_lr,
        finite_horizon_gae=args.finite_horizon_gae,
        detach_encoder_on_actor=args.detach_encoder_on_actor,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        lr_min_ratio=args.lr_min_ratio,
        actor_use_layer_norm=args.actor_use_layer_norm,
        value_use_layer_norm=args.value_use_layer_norm,
        actor_use_group_norm=args.actor_use_group_norm,
        value_use_group_norm=args.value_use_group_norm,
        num_groups=args.num_groups,
        actor_dropout_rate=args.actor_dropout_rate,
        value_dropout_rate=args.value_dropout_rate,
        kernel_init=args.kernel_init,
        backbone_type=args.backbone_type,
        log_std_init=args.log_std_init,
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
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
