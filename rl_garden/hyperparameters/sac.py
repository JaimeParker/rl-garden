"""Online SAC learning settings, independent of algorithm and entrypoint imports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import Activation, BackboneType, KernelInit


@dataclass
class SACParams(OfflineOptimizationArgs):
    """Constructor defaults for SAC."""

    nstep: int = 1
    tau: float = 0.01
    policy_lr: float = 0.0003
    q_lr: float = 0.0003
    alpha_lr: Optional[float] = None
    policy_frequency: int = 1
    target_network_frequency: int = 1
    ent_coef: float | str = 'auto'
    target_entropy: float | str = 'auto'
    alpha_tuning: Literal["legacy_exp", "log_alpha", "lagrange_softplus"] = 'legacy_exp'
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = None
    n_critics: int = 2
    critic_subsample_size: Optional[int] = None
    backup_entropy: bool = True
    critic_impl: Literal['vmap', 'legacy'] = 'vmap'
    actor_use_layer_norm: bool = False
    critic_use_layer_norm: bool = False
    actor_log_std_min: float = -5.0
    actor_log_std_mode: Literal['clamp', 'tanh'] = 'clamp'
    # Entropy-off (TD3-shaped) options of SAC's update: Gaussian noise of
    # this std, clipped to +-target_noise_clip, on the critic target's next
    # action (absolute action units; 0 draws nothing), and the critic
    # ensemble reduction in the actor loss.
    target_policy_noise: float = 0.0
    target_noise_clip: float = 0.5
    actor_q_agg: Literal["min", "mean", "first"] = 'min'


@dataclass
class SACFlowParams(SACParams):
    """Constructor defaults for SACFlow."""

    denoising_steps: int = 4
    noise_std: float = 0.3
    # None: three layers of 256 (the entry builds it from --flow-hidden-dim and
    # --flow-hidden-layers instead).
    flow_hidden_dims: Optional[tuple[int, ...]] = None
    flow_use_layer_norm: bool = False


@dataclass
class SequenceSACParams(SACParams):
    """Constructor defaults for SequenceSAC."""

    burn_in_len: int = 40
    learning_len: int = 40
    forward_len: int = 5
    prio_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6


@dataclass
class RecurrentSACParams(SequenceSACParams):
    """Constructor defaults for RecurrentSAC."""

    rnn_type: Literal["lstm", "gru"] = 'lstm'
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1


@dataclass
class TransformerSACParams(SequenceSACParams):
    """Constructor defaults for TransformerSAC."""

    embed_dim: int = 256
    head_dim: int = 64
    num_heads: int = 4
    num_transformer_layers: int = 3
    mlp_num: int = 2
    memory_len: int = 16
    dropout_rate: float = 0.0
    gru_bias: float = 2.0


@dataclass
class RLPDParams(SACParams):
    """Constructor defaults for RLPD."""

    n_critics: int = 10
    critic_subsample_size: Optional[int] = 2
    critic_use_layer_norm: bool = True
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: BackboneType = 'mlp'
    use_pnorm: bool = False
    exclude_bias_from_decay: bool = False
    std_parameterization: Literal['exp', 'uniform'] = 'exp'


@dataclass
class RLPDHybridParams(RLPDParams):
    """Constructor defaults for RLPDHybrid."""

    discrete_hidden_dim: int = 256
    discrete_lr: float = 0.0003
    discrete_tau: Optional[float] = None
    use_grasp_penalty: bool = False
    num_grippers: int = 1


@dataclass
class ExPLOReParams(RLPDParams):
    """Constructor defaults for ExPLORe."""

    offline_relabel_type: Literal["gt", "pred", "min"] = 'pred'
    use_rnd_offline: bool = False
    use_rnd_online: bool = False
    rnd_coeff: float = 1.0
    relabeler_hidden_dims: tuple[int, ...] = (256, 256)
    relabeler_lr: float = 0.0003
    relabeler_activation_fn: Optional[Activation] = None
    relabeler_kernel_init: Optional[KernelInit] = None
    rnd_hidden_dims: tuple[int, ...] = (256, 256)
    rnd_feature_dim: int = 256
    rnd_lr: float = 0.0003
    rnd_activation_fn: Optional[Activation] = None
    rnd_kernel_init: Optional[KernelInit] = None


@dataclass
class ACRLPDParams(RLPDParams):
    """Constructor defaults for ACRLPD."""

    horizon_length: int = 5
    target_entropy_multiplier: float = 0.5
    q_agg: Literal['mean', 'min'] = 'mean'
    bc_alpha: float = 0.0
    critic_subsample_size: Optional[int] = None
    backup_entropy: bool = False
