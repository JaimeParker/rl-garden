"""RGBD SAC on ManiSkill with pluggable image encoder.

Usage:
    python examples/train_sac_rgbd.py --env_id PickCube-v1 --encoder plain_conv
    python examples/train_sac_rgbd.py --env_id PickCube-v1 --encoder resnet10
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from torch.utils.tensorboard import SummaryWriter

from rl_garden.algorithms import RGBDSAC
from rl_garden.common import Logger, seed_everything
from rl_garden.encoders import default_image_encoder_factory, resnet_encoder_factory
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    env_id: str = "PickCube-v1"
    obs_mode: str = "rgb"  # "rgb" | "rgbd"
    include_state: bool = True
    num_envs: int = 16
    num_eval_envs: int = 16
    total_timesteps: int = 1_000_000
    buffer_size: int = 200_000
    buffer_device: str = "cuda"
    seed: int = 1
    batch_size: int = 512
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.25
    gamma: float = 0.8
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    encoder_features_dim: int = 256
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    control_mode: str = "pd_joint_delta_pos"
    log_freq: int = 1_000
    eval_freq: int = 25
    num_eval_steps: int = 50


def _image_factory(
    name: str,
    features_dim: int,
    pretrained_weights: Optional[str],
    freeze_resnet_encoder: bool,
    freeze_resnet_backbone: bool,
):
    if name == "plain_conv":
        if pretrained_weights is not None or freeze_resnet_encoder or freeze_resnet_backbone:
            raise ValueError(
                "--pretrained_weights, --freeze_resnet_encoder, and "
                "--freeze_resnet_backbone are only supported for resnet encoders."
            )
        return default_image_encoder_factory(features_dim=features_dim)
    return resnet_encoder_factory(
        name=name,
        features_dim=features_dim,
        pretrained_weights=pretrained_weights,
        freeze_resnet_encoder=freeze_resnet_encoder,
        freeze_resnet_backbone=freeze_resnet_backbone,
    )


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    run_name = (
        args.exp_name
        or f"{args.env_id}__sac_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    logger = Logger(tensorboard=writer)

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

    factory = _image_factory(
        args.encoder,
        args.encoder_features_dim,
        args.pretrained_weights,
        args.freeze_resnet_encoder,
        args.freeze_resnet_backbone,
    )
    image_keys = ("rgb",) if args.obs_mode == "rgb" else ("rgb", "depth")

    agent = RGBDSAC(
        env=env, eval_env=eval_env,
        buffer_size=args.buffer_size, buffer_device=args.buffer_device,
        learning_starts=args.learning_starts, batch_size=args.batch_size,
        gamma=args.gamma, tau=args.tau,
        training_freq=args.training_freq, utd=args.utd,
        policy_lr=args.policy_lr, q_lr=args.q_lr,
        seed=args.seed, logger=logger,
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        image_keys=image_keys,
        image_encoder_factory=factory,
    )
    agent.learn(total_timesteps=args.total_timesteps)

    logger.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
