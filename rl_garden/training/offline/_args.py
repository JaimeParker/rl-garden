"""Composable CLI arguments for offline algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.common.cli_args import CheckpointArgs, LoggingArgs
from rl_garden.common.env_args import EnvBackendArgs


@dataclass
class OfflineVisionArgs:
    """Vision settings inferred from an offline dataset rather than a live env."""

    obs_mode: str = "rgb"
    include_state: bool = True
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64
    encoder: Literal["plain_conv", "resnet10", "resnet18", "vit"] = "plain_conv"
    encoder_features_dim: int = 256
    image_fusion_mode: Literal["stack_channels", "per_key"] = "stack_channels"
    vit_fusion_mode: Literal["per_key", "stack_channels"] = "per_key"
    vit_embed_dim: int = 128
    vit_depth: int = 1
    vit_num_heads: int = 4
    vit_embed_norm: bool = False
    vit_augmentation: Literal["random_shift", "none"] = "random_shift"
    vit_random_shift_pad: int = 4
    vit_actor_feature_dim: int = 128
    vit_critic_spatial_emb_dim: int = 1024
    pretrained_weights: Optional[str] = None
    freeze_resnet_encoder: bool = False
    freeze_resnet_backbone: bool = False
    plain_conv_weight_init: Literal["kaiming_uniform", "orthogonal"] = "kaiming_uniform"
    plain_conv_last_act: bool = True
    plain_conv_pooling: Literal["flatten", "gap", "adaptive_max"] = "flatten"
    per_camera_rgbd: bool = False


@dataclass
class OfflineDatasetArgs:
    num_offline_steps: int = 100_000
    # "maniskill_h5": offline_dataset_path is a filesystem path to a ManiSkill
    # trajectory H5 file. "minari": offline_dataset_path is a Minari dataset id
    # instead (e.g. "D4RL/halfcheetah/medium-v2").
    dataset_source: Literal["maniskill_h5", "minari"] = "maniskill_h5"
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None
    save_filename: Optional[str] = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: Optional[str] = None
    action_low: float = -1.0
    action_high: float = 1.0
    spec_num_envs: int = 1


@dataclass
class OfflineReplayArgs:
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    offline_sampling: Literal["with_replace", "without_replace"] = "with_replace"


@dataclass
class OfflineOptimizationArgs:
    weight_decay: float = 0.0
    use_adamw: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None


@dataclass
class OfflineRuntimeArgs:
    seed: int = 1


@dataclass
class OfflineEvalArgs:
    env_id: Optional[str] = None
    num_eval_envs: int = 1
    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"


@dataclass
class OfflineCommonArgs(
    OfflineEvalArgs,
    OfflineRuntimeArgs,
    OfflineVisionArgs,
    OfflineOptimizationArgs,
    OfflineReplayArgs,
    OfflineDatasetArgs,
    LoggingArgs,
    CheckpointArgs,
    EnvBackendArgs,
):
    """Arguments shared by every offline algorithm."""


@dataclass
class TDMPC2MultitaskTrainingArgs(CheckpointArgs, LoggingArgs):
    """TD-MPC2 multitask offline pretraining.

    Deliberately does NOT inherit ``EnvRunArgs``/``EnvBackendArgs``/
    ``OfflineDatasetArgs``: there is no single ``env_id``/live env (training
    never touches one, see ``rl_garden.algorithms.tdmpc2.multitask.agent``)
    and no single homogeneous dataset (``dataset_dir`` points at the
    per-task, differently-shaped output of
    ``tools/conversion/convert_tdmpc2_multitask_dataset.py``, not one
    ``offline_dataset_path`` file).
    """

    dataset_dir: str = ""
    mmap_dir: str = ""
    device: str = "auto"
    num_offline_steps: int = 10_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    horizon: int = 3
    task_dim: int = 96
    latent_dim: int = 512
    enc_dim: int = 256
    num_enc_layers: int = 2
    mlp_dim: int = 512
    simnorm_dim: int = 8
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    dropout: float = 0.01
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    rho: float = 0.5
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995


@dataclass
class OfflineDeviceArgs:
    device: str = "auto"


@dataclass
class OfflineDiscountArgs:
    gamma: float = 0.99
    tau: float = 0.005
    utd: float = 1.0


@dataclass
class OfflineActorArgs:
    actor_use_layer_norm: bool = True
    actor_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    std_parameterization: Literal["exp", "uniform"] = "exp"


@dataclass
class OfflineCriticArgs:
    n_critics: int = 10
    critic_subsample_size: int = 2
    critic_use_layer_norm: bool = True
    critic_use_group_norm: bool = False
    critic_dropout_rate: Optional[float] = None


@dataclass
class OfflineValueArgs:
    value_use_layer_norm: bool = False
    value_use_group_norm: bool = False
    value_dropout_rate: Optional[float] = None


@dataclass
class OfflineCompileArgs:
    use_compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


@dataclass
class OfflineCQLArgs:
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    alpha_lr: float = 1e-4
    cql_alpha_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    use_cql_loss: bool = True
    use_td_loss: bool = True
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


@dataclass
class OfflineCalQLArgs:
    use_calql: bool = True
    calql_bound_random_actions: bool = False
    sparse_reward_mc: bool = False
    sparse_negative_reward: float = 0.0
    success_threshold: float = 0.5


@dataclass
class OfflineIQLArgs:
    actor_lr: float = 3e-4
    critic_value_lr: float = 3e-4
    expectile: float = 0.7
    temperature: float = 3.0
    adv_clip_max: float = 100.0


@dataclass
class OfflineBCArgs:
    actor_lr: float = 3e-4


@dataclass
class OfflineWSRLArgs:
    training_freq: int = 64
