"""Shared dataclass CLI arguments for training examples."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class LoggingArgs:
    log_dir: str = "runs"
    exp_name: Optional[str] = None
    log_freq: int = 1_000
    eval_freq: int = 10_000
    num_eval_steps: int = 50
    std_log: bool = True
    log_type: Literal["tensorboard", "wandb", "none"] = "wandb"
    log_keywords: Optional[str] = None
    wandb_project: str = "rl-garden"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None


@dataclass
class CheckpointArgs:
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 0
    load_checkpoint: Optional[str] = None
    save_replay_buffer: bool = False
    load_replay_buffer: bool = True
    save_final_checkpoint: bool = True


@dataclass
class ManiSkillRunArgs(LoggingArgs):
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    seed: int = 1
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"
    capture_video: bool = True
    video_fps: int = 30
    eval_output_dir: Optional[str] = None


@dataclass
class SACTrainingArgs(ManiSkillRunArgs, CheckpointArgs):
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 1024
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.5
    gamma: float = 0.8
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4


@dataclass
class VisionArgs:
    obs_mode: str = "rgb"
    include_state: bool = True
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal["plain_conv", "resnet10", "resnet18"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False


@dataclass
class VisionSACTrainingArgs(SACTrainingArgs, VisionArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class WSRLTrainingArgs(ManiSkillRunArgs, CheckpointArgs):
    num_offline_steps: int = 0
    num_online_steps: int = 1_000_000
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None
    online_replay_mode: Literal["empty", "append"] = "empty"

    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005

    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    alpha_lr: float = 1e-4
    cql_alpha_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1

    n_critics: int = 10
    critic_subsample_size: int = 2

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

    use_calql: bool = True
    calql_bound_random_actions: bool = False

    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = True
    std_parameterization: Literal["exp", "uniform"] = "exp"

    online_cql_alpha: Optional[float] = None
    online_use_cql_loss: Optional[bool] = None


@dataclass
class VisionWSRLTrainingArgs(WSRLTrainingArgs, VisionArgs):
    camera_width: Optional[int] = 128
    camera_height: Optional[int] = 128
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else None


def apply_log_env_overrides(args: LoggingArgs) -> None:
    args.std_log = _env_bool("RLG_STD_LOG", args.std_log)
    args.log_type = _env_str("RLG_LOG_TYPE", args.log_type) or args.log_type
    args.log_keywords = _env_str("RLG_LOG_KEYWORDS", args.log_keywords)
    args.wandb_project = (
        _env_str("RLG_WANDB_PROJECT", args.wandb_project) or args.wandb_project
    )
    args.wandb_entity = _env_str("RLG_WANDB_ENTITY", args.wandb_entity)
    args.wandb_group = _env_str("RLG_WANDB_GROUP", args.wandb_group)


def resolve_checkpoint_dir(args: Any, run_name: str) -> Optional[str]:
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    if not args.save_final_checkpoint and args.checkpoint_freq <= 0:
        return None
    return os.path.join(args.log_dir, run_name, "checkpoints")


def resolve_eval_record_dir(args: Any, run_name: str) -> str:
    if args.eval_output_dir:
        return args.eval_output_dir
    return os.path.join(args.log_dir, run_name, "eval_videos")


def image_encoder_factory_from_args(args: VisionArgs):
    if args.encoder == "plain_conv":
        if (
            args.pretrained_weights is not None
            or args.freeze_resnet_encoder
            or args.freeze_resnet_backbone
        ):
            raise ValueError(
                "--pretrained_weights, --freeze_resnet_encoder, and "
                "--freeze_resnet_backbone are only supported for resnet encoders."
            )
        from rl_garden.encoders import default_image_encoder_factory

        return default_image_encoder_factory(features_dim=args.encoder_features_dim)

    from rl_garden.encoders import resnet_encoder_factory

    return resnet_encoder_factory(
        name=args.encoder,
        features_dim=args.encoder_features_dim,
        pretrained_weights=args.pretrained_weights,
        freeze_resnet_encoder=args.freeze_resnet_encoder,
        freeze_resnet_backbone=args.freeze_resnet_backbone,
    )


def image_keys_from_obs_mode(obs_mode: str) -> tuple[str, ...]:
    return ("rgb",) if obs_mode == "rgb" else ("rgb", "depth")
