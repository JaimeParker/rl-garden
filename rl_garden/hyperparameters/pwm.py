"""Model-based learning settings shared by algorithms and CLI Args."""
from dataclasses import dataclass


@dataclass
class PWMParams:
    """Constructor settings for PWM."""

    horizon: int = 16
    lam: float = 0.95
    actor_lr: float = 0.0005
    critic_lr: float = 0.0005
    lr_final: float = 1e-05
    actor_grad_norm: float = 1.0
    critic_grad_norm: float = 100.0
    critic_iterations: int = 8
    critic_batches: int = 4
    num_critics: int = 3
    actor_units: tuple[int, ...] = (400, 200, 100)
    critic_units: tuple[int, ...] = (400, 200)
    init_logstd: float = -1.0
    min_logstd: float = -1.427
    ret_rms: bool = True
