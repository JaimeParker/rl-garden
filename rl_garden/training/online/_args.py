"""CLI argument dataclasses for online training algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.common.cli_args import CheckpointArgs, LoggingArgs, VisionArgs
from rl_garden.common.env_args import EnvRunArgs
from rl_garden.common.training_phase import InitialTrainingPhase


@dataclass
class SACTrainingArgs(EnvRunArgs, CheckpointArgs):
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 1024
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.5
    gamma: float = 0.8
    nstep: int = 1
    tau: float = 0.01
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    critic_impl: Literal["vmap", "legacy"] = "vmap"
    n_critics: int = 2
    critic_subsample_size: Optional[int] = None
    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = False
    hidden_dim: int = 256
    actor_hidden_layers: int = 3
    critic_hidden_layers: int = 3
    actor_log_std_min: float = -5.0
    actor_log_std_mode: Literal["clamp", "tanh"] = "clamp"
    alpha_tuning: Literal["legacy_exp", "log_alpha", "lagrange_softplus"] = "legacy_exp"
    ent_coef: float | str = "auto"
    target_entropy: float | str = "auto"
    alpha_lr: Optional[float] = None
    q_landscape_diagnostics: bool = False
    q_landscape_num_actions: int = 8
    q_landscape_batch_size: int = 64
    q_mc_diagnostics: bool = False
    critic_only_steps: int = 0
    critic_only_freeze_encoder: bool = True
    critic_only_random_action_prob: float = 0.0
    load_actor_checkpoint: Optional[str] = None


@dataclass
class VisionSACTrainingArgs(SACTrainingArgs, VisionArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class RecurrentSACTrainingArgs(SACTrainingArgs):
    rnn_type: Literal["lstm", "gru"] = "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    burn_in_len: int = 40
    learning_len: int = 40
    forward_len: int = 5
    prio_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6


@dataclass
class VisionRecurrentSACTrainingArgs(RecurrentSACTrainingArgs, VisionArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class PPOTrainingArgs(EnvRunArgs, CheckpointArgs):
    total_timesteps: int = 10_000_000
    num_steps: int = 50
    gamma: float = 0.8
    gae_lambda: float = 0.9
    learning_rate: float = 3e-4
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    anneal_lr: bool = False
    finite_horizon_gae: bool = False
    detach_encoder_on_actor: Optional[bool] = None
    weight_decay: float = 0.0
    use_adamw: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    actor_use_layer_norm: bool = False
    value_use_layer_norm: bool = False
    actor_use_group_norm: bool = False
    value_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    value_dropout_rate: Optional[float] = None
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    log_std_init: float = -0.5


@dataclass
class VisionPPOTrainingArgs(PPOTrainingArgs, VisionArgs):
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64


@dataclass
class RecurrentPPOTrainingArgs(PPOTrainingArgs):
    rnn_type: Literal["lstm", "gru"] = "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1


@dataclass
class VisionRecurrentPPOTrainingArgs(RecurrentPPOTrainingArgs, VisionArgs):
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64


@dataclass
class TransformerPPOTrainingArgs(PPOTrainingArgs):
    embed_dim: int = 256
    head_dim: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 3
    mlp_num: int = 2
    memory_len: int = 64
    dropout_rate: float = 0.0
    gru_bias: float = 2.0


@dataclass
class VisionTransformerPPOTrainingArgs(TransformerPPOTrainingArgs, VisionArgs):
    camera_width: Optional[int] = 64
    camera_height: Optional[int] = 64


@dataclass
class DrQv2TrainingArgs:
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 16
    obs_mode: str = "rgbd"
    include_state: bool = True
    control_mode: str = "pd_ee_delta_pose"
    camera_width: int = 128
    camera_height: int = 128
    render_mode: str = "rgb_array"
    per_camera_rgbd: bool = False
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    mmap_dir: Optional[str] = None
    mmap_mode: Literal["create", "open"] = "create"
    learning_starts: int = 4_000
    batch_size: int = 256
    seed: int = 1
    gamma: float = 0.99
    tau: float = 0.01
    training_freq: int = 32
    utd: float = 0.5
    policy_lr: float = 1e-4
    q_lr: float = 1e-4
    feature_dim: int = 50
    hidden_dim: int = 1024
    nstep: int = 3
    stddev_schedule: str = "linear(1.0,0.1,500000)"
    actor_stddev_schedule: Optional[str] = None
    stddev_clip: float = 0.3
    num_expl_steps: int = 2000
    grad_clip_norm: Optional[float] = None
    weight_decay: float = 0.0
    use_adamw: bool = False
    image_fusion_mode: str = "stack_channels"
    image_augmentation: str = "random_shift"
    image_random_shift_pad: int = 4
    frame_stack: int = 1
    log_type: str = "wandb"
    log_dir: str = "runs"
    exp_name: str = ""
    wandb_project: str = "rl-garden"
    wandb_entity: str = ""
    wandb_group: str = ""
    log_keywords: str = ""
    std_log: bool = True
    log_freq: int = 1_000
    eval_freq: int = 10_000
    num_eval_steps: int = 50
    capture_video: bool = True
    eval_output_dir: Optional[str] = None
    video_fps: int = 30
    checkpoint_dir: Optional[str] = None
    checkpoint_freq: int = 0
    save_replay_buffer: bool = False
    save_final_checkpoint: bool = True
    load_checkpoint: Optional[str] = None
    load_replay_buffer: bool = False
    replay_lazy_next_obs: bool = False
    replay_pin_sampled_batch: bool = False


@dataclass
class FlashSACTrainingArgs(LoggingArgs):
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 8
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"
    buffer_size: int = 10_000_000
    buffer_device: str = "cuda"
    learning_starts: int = 4_000
    batch_size: int = 2048
    gamma: float = 0.99
    tau: float = 0.005
    training_freq: int = 512
    utd: float = 1.0
    n_step: int = 3
    total_timesteps: int = 10_000_000
    actor_hidden_dim: int = 128
    actor_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_blocks: int = 2
    num_bins: int = 101
    min_v: float = -5.0
    max_v: float = 5.0
    asymmetric_obs_dim: int = 0
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    actor_update_period: int = 1
    grad_clip_norm: Optional[float] = None
    temp_initial_value: float = 0.01
    target_entropy: str = "auto"
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    normalize_reward: bool = False
    normalized_g_max: float = 10.0
    bc_alpha: float = 0.0
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    capture_video: bool = False
    video_fps: int = 20
    seed: int = 1
    checkpoint_freq: int = 0
    save_replay_buffer: bool = False
    save_final_checkpoint: bool = True
    load_checkpoint: Optional[str] = None


def sac_initial_training_phase_from_args(
    args: SACTrainingArgs,
) -> Optional[InitialTrainingPhase]:
    if args.critic_only_steps <= 0:
        return None
    return InitialTrainingPhase(
        duration_steps=args.critic_only_steps,
        update_actor=False,
        update_critic=True,
        update_encoder=not args.critic_only_freeze_encoder,
        random_action_prob=args.critic_only_random_action_prob,
    )
