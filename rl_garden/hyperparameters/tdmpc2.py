"""Model-based learning settings shared by algorithms and CLI Args."""
from dataclasses import dataclass


@dataclass
class TDMPC2CoreParams:
    """Constructor settings for TDMPC2Core."""

    horizon: int = 3
    latent_dim: int = 512
    mlp_dim: int = 512
    simnorm_dim: int = 8
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    dropout: float = 0.01
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 0.0001
    lr: float = 0.0003
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
class TDMPC2Params(TDMPC2CoreParams):
    """Constructor settings for TDMPC2."""

    use_planner: bool = True
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    iterations: int = 6
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5


@dataclass
class TDMPC2MultitaskParams(TDMPC2CoreParams):
    """Constructor settings for TDMPC2Multitask."""

    task_dim: int = 96
    enc_dim: int = 256
    num_enc_layers: int = 2
