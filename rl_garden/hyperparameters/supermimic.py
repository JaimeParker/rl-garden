"""Model-based learning settings shared by algorithms and CLI Args."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class SuperMimicParams:
    """Constructor settings for SuperMimic."""

    heads: Literal['learned', 'analytic'] = 'learned'
    replay_ratio: Optional[float] = 16.0
    wm_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    head_hidden_dims: tuple[int, ...] = (256, 256)
    latent_dim: int = 32
    goal_encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_dims: tuple[int, ...] = (256, 128)
    horizon: int = 32
    wm_window: int = 7
    wm_batches: int = 10
    wm_batch_size: int = 2048
    imagination_batch: int = 4096
    lam: float = 0.95
    max_step_reward: Optional[float] = None
    lr: float = 0.0002
    lr_final_ratio: float = 1.0
    lr_decay_updates: int = 250000
    actor_betas: tuple[float, float] = (0.9, 0.95)
    wm_grad_norm: float = 1.0
    actor_grad_norm: float = 1.0
    actor_grad_value: float = 0.1
    critic_grad_norm: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise_std: float = 0.1
    obs_history_len: int = 1
    warmup_updates: int = 0
