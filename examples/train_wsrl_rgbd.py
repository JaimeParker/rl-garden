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

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from rl_garden.algorithms import WSRLRGBD
from rl_garden.buffers import load_maniskill_h5_to_replay_buffer
from rl_garden.common import Logger, seed_everything
from rl_garden.encoders import default_image_encoder_factory, resnet_encoder_factory
from rl_garden.envs import ManiSkillEnvConfig, make_maniskill_env


@dataclass
class Args:
    # Environment
    env_id: str = "PickCube-v1"
    obs_mode: str = "rgb"  # "rgb" | "rgbd"
    include_state: bool = True
    num_envs: int = 16
    num_eval_envs: int = 16
    control_mode: str = "pd_joint_delta_pos"
    camera_width: Optional[int] = 128
    camera_height: Optional[int] = 128

    # Training phases
    num_offline_steps: int = 0
    num_online_steps: int = 1_000_000
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None

    # Buffer and training
    buffer_size: int = 200_000
    buffer_device: str = "cuda"
    seed: int = 1
    batch_size: int = 512
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.25
    gamma: float = 0.99
    tau: float = 0.005

    # Optimizers
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    alpha_lr: float = 1e-4
    cql_alpha_lr: float = 3e-4

    # Encoder
    encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    encoder_features_dim: int = 256
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False

    # Q-ensemble (REDQ)
    n_critics: int = 10
    critic_subsample_size: int = 2

    # CQL parameters
    use_cql_loss: bool = True
    cql_n_actions: int = 10
    cql_action_sample_method: Literal["uniform", "normal"] = "uniform"
    cql_alpha: float = 5.0
    cql_autotune_alpha: bool = False
    cql_alpha_lagrange_init: float = 1.0
    cql_target_action_gap: float = 1.0
    cql_importance_sample: bool = True
    cql_max_target_backup: bool = True
    cql_temp: float = 1.0
    cql_clip_diff_min: float = float("-inf")
    cql_clip_diff_max: float = float("inf")
    backup_entropy: bool = False

    # Cal-QL parameters
    use_calql: bool = True
    calql_bound_random_actions: bool = False

    # Upstream WSRL network options
    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = True
    std_parameterization: Literal["exp", "uniform"] = "exp"

    # Phase control
    online_cql_alpha: Optional[float] = None
    online_use_cql_loss: Optional[bool] = None

    # Logging
    log_dir: str = "runs"
    exp_name: Optional[str] = None
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


def _offline_update_loop(agent: WSRLRGBD, steps: int, logger: Logger, log_freq: int) -> None:
    gradient_steps = int(agent.utd) if float(agent.utd).is_integer() and agent.utd > 1 else 1
    for step in trange(steps, desc="offline"):
        losses = agent.train(gradient_steps)
        if log_freq > 0 and (step + 1) % log_freq == 0:
            for key, value in losses.items():
                logger.add_scalar(f"offline_losses/{key}", value, step + 1)


def main() -> None:
    args = tyro.cli(Args)
    seed_everything(args.seed)

    run_name = (
        args.exp_name
        or f"{args.env_id}__wsrl_rgbd_{args.encoder}__{args.seed}__{int(time.time())}"
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
        log_freq=args.log_freq, eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        image_keys=image_keys,
        image_encoder_factory=factory,
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
        _offline_update_loop(agent, args.num_offline_steps, logger, args.log_freq)

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
