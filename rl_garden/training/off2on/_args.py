"""CLI argument dataclasses for offline-to-online training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import warnings

from rl_garden.common.cli_args import CheckpointArgs, VisionArgs
from rl_garden.common.env_args import EnvRunArgs
from rl_garden.common.training_phase import InitialTrainingPhase


@dataclass
class WSRLTrainingArgs(EnvRunArgs, CheckpointArgs):
    num_offline_steps: int = 0
    num_online_steps: int = 1_000_000
    offline_dataset_path: Optional[str] = None
    offline_num_traj: Optional[int] = None
    online_replay_mode: Literal["empty", "append", "mixed"] = "empty"
    offline_data_ratio: float = 0.0
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 4.0
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    alpha_lr: float = 1e-4
    cql_alpha_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    weight_decay: float = 0.0
    use_adamw: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
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
    actor_use_group_norm: bool = False
    critic_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    std_parameterization: Literal["exp", "uniform"] = "exp"
    online_cql_alpha: float = 0.0
    online_use_cql_loss: bool = False
    warmup_steps: int = 5000
    offline_sampling: Literal["with_replace", "without_replace"] = "with_replace"
    sparse_reward_mc: bool = False
    sparse_negative_reward: float = 0.0
    success_threshold: float = 0.5
    success_key: Optional[str] = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0


@dataclass
class VisionWSRLTrainingArgs(WSRLTrainingArgs, VisionArgs):
    camera_width: Optional[int] = 128
    camera_height: Optional[int] = 128
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


def wsrl_initial_training_phase_from_args(
    args: WSRLTrainingArgs,
) -> Optional[InitialTrainingPhase]:
    if args.warmup_steps <= 0:
        return None
    return InitialTrainingPhase(
        duration_steps=args.warmup_steps,
        update_actor=False,
        update_critic=False,
        update_encoder=False,
        random_action_prob=0.0,
    )


def warn_if_wsrl_warmup_uses_uninitialized_policy(args: WSRLTrainingArgs) -> None:
    if args.warmup_steps > 0 and args.load_checkpoint is None:
        warnings.warn(
            "warmup_steps > 0 without an offline-trained checkpoint will use "
            "the randomly initialized policy to collect data while updates are "
            "paused. Use --load_checkpoint or set --warmup_steps 0 unless this "
            "is intentional.",
            UserWarning,
            stacklevel=2,
        )
