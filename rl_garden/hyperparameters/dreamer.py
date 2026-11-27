"""Learning inputs for Dreamer and reward-free RSSM pretraining."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class RSSMArchitectureParams:
    stoch: int = 32
    unimix: float = 0.01
    blocks: int = 8
    obs_layers: int = 1
    img_layers: int = 2
    dyn_layers: int = 1
    decoder_layers: int = 3


@dataclass
class RSSMLossParams:
    kl_free: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    recon_scale: float = 1.0
    batch_length: int = 64


@dataclass
class DreamerCoreParams(RSSMLossParams):
    reward_bins: int = 255
    train_ratio: float = 512.0
    imag_horizon: int = 15
    lam: float = 0.95
    slow_target_fraction: float = 0.02


@dataclass
class DreamerV3Params(DreamerCoreParams, RSSMArchitectureParams):
    size: Literal["12M", "25M", "50M", "100M", "200M", "400M", "ncrl_m"] = "12M"
    # contdisc: official JAX default is True (continue-head target
    # multiplied by 1 - 1/horizon, imagination disc=1); r2dreamer has no
    # such switch (target 1 - is_terminal, disc=1-1/horizon applied
    # outside). This port"s default (False) matches r2dreamer -- see
    # RSSM"s own docstring.
    contdisc: bool = False
    discount_horizon: float = 333.0
    act_entropy: float = 0.0003
    rew_scale: float = 1.0
    con_scale: float = 1.0
    policy_scale: float = 1.0
    value_scale: float = 1.0
    repval_scale: float = 0.3
    lr: float = 4e-05
    warmup: int = 1000


@dataclass
class NCRLParams(DreamerCoreParams):
    train_ratio: float = 256.0
    imag_horizon: int = 16
    online_ratio: float = 0.75
    num_retrieve_episodes: int = 500
    max_retrieve_iterations: int = 10
    use_guidance: bool = True
    bc_train_steps: int = 150_000
    bc_batch_size: int = 256
    bc_lr: float = 0.0001
    guidance_schedule: tuple[str, float, float, int] = ("linear", 1.0, 0.1, 100_000)
    warmup_frames: int = 15_000
    discount: float = 0.99
    actor_ent: float = 0.0001
    model_lr: float = 0.0001
    reward_lr: float = 0.0001
    actor_lr: float = 8e-05
    critic_lr: float = 8e-05
    adam_eps: float = 1e-05
    weight_decay: float = 1e-06
    grad_clip: float = 100.0
    ac_units: int = 512
    actor_layers: int = 3
    critic_layers: int = 3
    reward_layers: int = 2
    reward_units: int = 512
    kl_free_mode: Literal["per_element", "mean"] = "mean"


@dataclass
class RSSMPretrainParams(RSSMLossParams, RSSMArchitectureParams):
    backbone: Literal["dreamer_v3", "ncrl"] = "dreamer_v3"
    size: str = "ncrl_m"
    deter: Optional[int] = None
    hidden: Optional[int] = None
    classes: Optional[int] = None
    cnn_depth: Optional[int] = None
    kl_free_mode: Literal["per_element", "mean"] = "mean"
    lr: float = 0.0001
    adam_eps: float = 1e-05
    weight_decay: float = 1e-06
    grad_clip: float = 100.0

