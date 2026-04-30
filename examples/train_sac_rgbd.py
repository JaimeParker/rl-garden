"""RGBD SAC on ManiSkill with pluggable image encoder.

Usage:
    python examples/train_sac_rgbd.py --env_id PickCube-v1 --encoder plain_conv
    python examples/train_sac_rgbd.py --env_id PickCube-v1 --encoder resnet10
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import tyro

from rl_garden.algorithms import RGBDSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.common.cli_args import (
    VisionSACTrainingArgs,
    apply_log_env_overrides,
    image_encoder_factory_from_args,
    image_keys_from_obs_mode,
)
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args(VisionSACTrainingArgs):
    pass


def main() -> None:
    args = tyro.cli(Args)
    apply_log_env_overrides(args)
    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name
        or f"{args.env_id}__sac_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
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
    )
    eval_cfg = ManiSkillEnvConfig(
        env_id=args.env_id, num_envs=args.num_eval_envs,
        obs_mode=args.obs_mode, include_state=args.include_state,
        control_mode=args.control_mode, reconfiguration_freq=1,
        camera_width=args.camera_width, camera_height=args.camera_height,
    )
    env = make_maniskill_env(env_cfg)
    eval_env = make_maniskill_env(eval_cfg)

    factory = image_encoder_factory_from_args(args)
    image_keys = image_keys_from_obs_mode(args.obs_mode)

    agent = RGBDSAC(
        env=env, eval_env=eval_env,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, tau=args.tau,
        training_freq=args.training_freq, utd=args.utd,
        policy_lr=args.policy_lr, q_lr=args.q_lr,
        seed=args.seed, logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        image_keys=image_keys,
        image_encoder_factory=factory,
        image_fusion_mode=args.image_fusion_mode,
    )
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
