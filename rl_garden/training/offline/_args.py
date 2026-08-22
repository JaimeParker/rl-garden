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
    # The dataset locator is a filesystem path for "h5", a dataset id for
    # "minari", and a legacy Gym environment id for "d4rl_legacy".
    dataset_backend: Literal["h5", "minari", "d4rl_legacy"] = "h5"
    offline_dataset: Optional[str] = None
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
    num_eval_episodes: int = 100
    num_eval_steps: Optional[int] = None
    # Expected worst-case env steps for one eval episode (e.g. 1000 for
    # AntMaze). Only sizes the eval step budget when num_eval_steps is unset
    # (num_eval_steps = num_eval_episodes * eval_episode_horizon) -- does not
    # impose a TimeLimit on the eval env itself.
    eval_episode_horizon: Optional[int] = None
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
    ``offline_dataset`` file).
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
class DiffusionBCTrainingArgs(CheckpointArgs, LoggingArgs):
    """Diffusion BC pretraining (DPPO phase 1). Deliberately does NOT inherit
    ``OfflineCommonArgs``: ``run_offline`` assumes a ``agent.replay_buffer``
    populated via ``load_offline_dataset``, but ``DiffusionBC`` loads
    ``(obs_history, action_chunk)`` windows directly in its constructor (see
    ``rl_garden.buffers.chunked_dataset``) and has no replay buffer at all --
    same reasoning as ``TDMPC2MultitaskTrainingArgs``."""

    dataset_path: str = ""
    num_offline_steps: int = 200_000
    offline_num_traj: Optional[int] = None
    horizon_steps: int = 4
    cond_steps: int = 1
    denoising_steps: int = 20
    activation_fn: Literal["relu", "gelu", "mish"] = "relu"
    residual_style: bool = True
    time_dim: int = 16
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    denoised_clip_value: Optional[float] = 1.0
    randn_clip_value: float = 10.0
    final_action_clip_value: Optional[float] = None
    min_sampling_denoising_std: float = 0.1
    actor_lr: float = 1e-3
    weight_decay: float = 1e-6
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
    batch_size: int = 128
    ema_decay: float = 0.995
    ema_update_every: int = 10
    ema_start_step: int = 0
    seed: int = 1
    device: str = "auto"


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
        Literal[
            "xavier_uniform",
            "xavier_normal",
            "orthogonal",
            "kaiming_uniform",
            "orthogonal_near_zero_output",
        ]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    std_parameterization: Literal["exp", "uniform"] = "exp"


@dataclass
class OfflineSACNetworkArgs:
    hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 4


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
    cql_penalty_scale: Literal["lagrange_only", "lagrange_times_alpha"] = "lagrange_only"
    cql_diff_clip_mode: Literal["skip_when_autotune", "always"] = "skip_when_autotune"
    cql_alpha_param: Literal["softplus", "exp_clip"] = "softplus"
    backup_entropy: bool = False
    policy_log_std_multiplier: Optional[float] = None
    policy_log_std_offset: Optional[float] = None


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
    actor_distribution: Literal["squashed", "unsquashed"] = "squashed"
    actor_lr_schedule: Optional[Literal["constant", "linear_warmup", "warmup_cosine"]] = None
    actor_lr_warmup_steps: Optional[int] = None
    actor_lr_decay_steps: Optional[int] = None
    actor_lr_min_ratio: Optional[float] = None


@dataclass
class OfflineBCArgs:
    actor_lr: float = 3e-4


@dataclass
class OfflineWSRLArgs:
    training_freq: int = 64


@dataclass
class OfflineDeterministicActorCriticArgs:
    """Network toggles shared by TD3-BC and AWAC (2 fixed critics, no ensemble
    subsampling -- unlike ``OfflineActorArgs``/``OfflineCriticArgs``, which are
    tuned for CQL/IQL's larger critic ensembles)."""

    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = False
    actor_use_group_norm: bool = False
    critic_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = None
    backbone_type: Literal["mlp", "mlp_resnet"] = "mlp"
    n_critics: int = 2


@dataclass
class OfflineTD3BCArgs(OfflineDeterministicActorCriticArgs):
    """TD3-BC hyperparameters. Defaults match CORL's ``TrainConfig``."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5


@dataclass
class OfflineReBRACArgs(OfflineDeterministicActorCriticArgs):
    """ReBRAC hyperparameters. Defaults match CORL's ``rebrac.py::Config``."""

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    tau: float = 5e-3
    # CORL's actor_n_hiddens/critic_n_hiddens=3 (hidden_dim=256 each) --
    # TD3BC's own CLI args have no net_arch field at all (never wired
    # through, letting ReBRAC.__init__'s own None -> [256,256] default
    # apply); ReBRAC needs the 3-layer width explicitly.
    net_arch: tuple[int, ...] = (256, 256, 256)
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    normalize_q: bool = True
    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = True


@dataclass
class OfflineAWACArgs(OfflineDeterministicActorCriticArgs):
    """AWAC hyperparameters. Defaults match CORL's ``TrainConfig``."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    awac_lambda: float = 1.0
    exp_adv_max: float = 100.0


@dataclass
class OfflineFQLArgs(OfflineDeterministicActorCriticArgs):
    """FQL hyperparameters. Defaults match FQL's ``get_config``."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_use_layer_norm: bool = True
    alpha: float = 10.0
    flow_steps: int = 10
    q_agg: Literal["mean", "min"] = "mean"
    normalize_q_loss: bool = False
    # FQL's reference applies its default_init() (Xavier-uniform, zero bias)
    # unconditionally to every nn.Dense -- overrides
    # OfflineDeterministicActorCriticArgs's None default (TD3-BC/AWAC's
    # PyTorch-native references have no such fixed-init convention).
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = "xavier_uniform"
    # FQL's reference hardcodes nn.gelu unconditionally in every MLP --
    # overrides OfflineDeterministicActorCriticArgs's implicit ReLU default
    # (no field there today; every other algorithm in the codebase has none).
    activation_fn: Optional[Literal["relu", "gelu"]] = "gelu"
    # "shared": one encoder (AGENTS.md's project convention, matches SACPolicy).
    # "separate": three independent encoder instances, matching FQL's own
    # JAX reference. Only meaningful for Dict (vision) observation spaces --
    # Box observations use a parameterless FlattenExtractor either way.
    encoder_sharing: Literal["shared", "separate"] = "shared"


@dataclass
class OfflineQGFArgs(OfflineDeterministicActorCriticArgs):
    """QGF (Q-Guided Flow) hyperparameters. Defaults match qgf's get_config().
    State-only (Box observations) -- no vision support in v1."""

    horizon_length: int = 1
    actor_lr: float = 3e-4
    critic_value_lr: float = 3e-4
    # QGF's reference applies use_layer_norm=1 to every network.
    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = True
    value_use_layer_norm: bool = True
    expectile: float = 0.9
    q_agg: Literal["mean", "min"] = "min"
    denoise_steps: int = 10
    # "grid" matches QGF's own policy_loss (qgf.py:69-74, discrete-grid t);
    # "uniform" reproduces IFQL's own actor loss (ifql.py:72) instead.
    t_sampling: Literal["grid", "uniform"] = "grid"
    sampling_mode: Literal["guided", "grad_step", "best_of_n", "bptt", "robust_q"] = "guided"
    guidance_weight: float = 1.0
    denoised_action_approx: Literal["one_euler_step_approx", "noisy"] = (
        "one_euler_step_approx"
    )
    qgrad_step_size: float = 0.1
    qgrad_steps: int = 1
    use_sign_gradient: bool = False
    actor_num_samples: int = 32
    # RobustQ (sampling_mode="robust_q") only -- both reconstructed from
    # upstream-broken code, see qgf_policy.py's docstring.
    robust_critic_lr: float = 3e-4
    robust_critic_t_emb_size: int = 16
    # QGF's reference applies default_init() (Xavier-uniform-like) unconditionally.
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = "xavier_uniform"
    activation_fn: Optional[Literal["relu", "gelu"]] = "gelu"


@dataclass
class OfflineQAMArgs(OfflineDeterministicActorCriticArgs):
    """QAM (Q-learning with Adjoint Matching) hyperparameters. Defaults
    match qam's get_config(). State-only (Box observations) -- no vision
    support in v1. `edit_scale`'s network construction is a best-effort
    reconstruction of upstream-missing code (see rl_garden/policies/
    qam_policy.py's docstring)."""

    horizon_length: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip_norm: Optional[float] = 1.0
    # QAM's reference has one value_layer_norm=True knob shared by both
    # critic and value nets, and actor_layer_norm=False -- matches here via
    # separate critic_use_layer_norm/value_use_layer_norm set to the same
    # value.
    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = True
    value_use_layer_norm: bool = True
    critic_loss_type: Literal["ddpg", "iql"] = "ddpg"
    rho: float = 0.0
    expectile: float = 0.9
    flow_steps: int = 10
    best_of_n: int = 1
    inv_temp: float = 0.3
    residual: bool = False
    target_actor: bool = True
    clip_adj: bool = True
    use_target_grad: bool = True
    fql_alpha: float = 0.0
    edit_scale: float = 0.0
    edit_target_entropy: Optional[float] = None
    edit_target_entropy_multiplier: float = 0.5
    edit_alpha_lr: float = 3e-4
    # QAM's reference applies default_init() (Xavier-uniform-like) unconditionally.
    kernel_init: Optional[
        Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
    ] = "xavier_uniform"
    activation_fn: Optional[Literal["relu", "gelu"]] = "gelu"


@dataclass
class OfflineEDACArgs:
    """EDAC hyperparameters. Defaults match CORL's ``edac.py::TrainConfig``.
    Field names follow ``OfflineSAC``'s own convention (``policy_lr``/
    ``q_lr``/``alpha_lr``, not TD3-BC-family's ``actor_lr``/``critic_lr``),
    since ``EDAC`` subclasses ``OfflineSAC`` directly."""

    tau: float = 5e-3
    eta: float = 1.0
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: Optional[float] = 3e-4
    weight_decay: float = 0.0
    use_adamw: bool = False
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: Optional[float] = None
    ent_coef: str = "auto"
    target_entropy: str = "auto"
    # CORL's Actor/VectorizedCritic both use 3 hidden layers of width 256.
    net_arch: tuple[int, ...] = (256, 256, 256)
    n_critics: int = 10
    critic_subsample_size: Optional[int] = None


@dataclass
class OfflineSPOTArgs(OfflineDeterministicActorCriticArgs):
    """SPOT hyperparameters. Defaults match CORL's ``spot.py::TrainConfig``."""

    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    vae_lr: float = 1e-3
    vae_hidden_dim: int = 750
    vae_latent_dim: Optional[int] = None
    vae_iterations: int = 100_000
    beta: float = 0.5
    lambd: float = 1.0
    num_samples: int = 1
    iwae: bool = False
    lambd_cool: bool = False
    lambd_end: float = 0.2
