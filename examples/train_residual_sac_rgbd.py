"""Residual SAC on ManiSkill Dict RGBD observations.

Usage:
    python examples/train_residual_sac_rgbd.py --debug --env_id PickCube-v1
    python examples/train_residual_sac_rgbd.py --debug --env_id PickCube-v1 --encoder resnet10

Default mode uses an ACT base-action provider. Debug mode uses a zero provider,
which exercises the residual rollout/update path without checkpoint loading.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import tyro

from rl_garden.algorithms import ResidualSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionSACTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
    resolve_checkpoint_dir,
    resolve_eval_record_dir,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


class ZeroBaseActionProvider:
    """Base-action provider that always returns raw env-space zeros."""

    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self.action_shape = action_shape

    def __call__(self, obs):
        if isinstance(obs, dict):
            first = next(iter(obs.values()))
            num_envs = first.shape[0]
            device = first.device
        else:
            num_envs = obs.shape[0]
            device = obs.device
        return torch.zeros((num_envs,) + self.action_shape, device=device)

    def reset(self, env_ids=None) -> None:
        del env_ids


@dataclass
class ResidualRGBDArgs(VisionSACTrainingArgs):
    residual_action_scale: float = 0.1
    debug: bool = False
    policy: Literal["act", "zero"] = "act"
    ckpt_path: Optional[str] = "act-peg-only"
    act_temporal_agg: bool = True
    act_temporal_agg_k: float = 0.01


def make_base_action_provider(args: ResidualRGBDArgs, env):
    action_shape = tuple(env.single_action_space.shape)
    if args.debug or args.policy == "zero":
        return ZeroBaseActionProvider(action_shape)
    if args.policy == "act":
        from rl_garden.models.act import ACTBaseActionProvider

        provider = ACTBaseActionProvider.from_checkpoint(
            observation_space=env.single_observation_space,
            action_space=env.single_action_space,
            ckpt_path=args.ckpt_path,
            env=env,
            temporal_agg=args.act_temporal_agg,
            temporal_agg_k=args.act_temporal_agg_k,
        )
        print(
            "[residual] base_policy=act "
            f"ckpt={provider.checkpoint_path} "
            f"state_dim={provider.spec.state_dim} "
            f"action_dim={provider.spec.action_dim} "
            f"num_queries={provider.config.num_queries}",
            flush=True,
        )
        return provider
    raise ValueError(f"Unsupported residual base policy: {args.policy!r}.")


def run_residual_rgbd_training(
    args: ResidualRGBDArgs,
    env_cfg: ManiSkillEnvConfig,
    eval_cfg: ManiSkillEnvConfig,
    *,
    run_label: str,
) -> None:
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = "debug_zero_base" if args.debug else f"{args.policy}_base"
    run_name = (
        args.exp_name
        or f"{args.env_id}__{run_label}_{suffix}_{args.encoder}__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = resolve_checkpoint_dir(args, run_name)
    eval_cfg.record_dir = resolve_eval_record_dir(args, run_name)
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

    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)

    try:
        base_action_provider = make_base_action_provider(args, env)
        agent = ResidualSAC(
            env=env,
            eval_env=eval_env,
            base_action_provider=base_action_provider,
            residual_action_scale=args.residual_action_scale,
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
        agent.learn(total_timesteps=args.total_timesteps)
    finally:
        logger.close()
        env.close()
        eval_env.close()


def main() -> None:
    args = tyro.cli(ResidualRGBDArgs)
    env_cfg = ManiSkillEnvConfig(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        include_state=args.include_state,
        control_mode=args.control_mode,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        render_mode=args.render_mode,
    )
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
        save_video=args.capture_video,
        video_fps=args.video_fps,
        max_steps_per_video=args.num_eval_steps,
    )
    run_residual_rgbd_training(
        args,
        env_cfg,
        eval_cfg,
        run_label="residual_sac_rgbd",
    )


if __name__ == "__main__":
    main()
