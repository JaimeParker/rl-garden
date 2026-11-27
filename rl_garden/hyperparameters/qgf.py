"""QGF learning hyperparameters shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType


@dataclass
class QGFParams(OfflineOptimizationArgs):
    """Defaults of the QGF learning algorithm."""

    # Target critic Polyak update coefficient.
    tau: float = 0.005
    # Actor learning rate.
    actor_lr: float = 3e-4
    # Joint critic and value learning rate.
    critic_value_lr: float = 3e-4
    # Expectile of the value regression.
    expectile: float = 0.9
    # Critic ensemble aggregation for value targets and guidance.
    q_agg: Literal["mean", "min"] = "min"
    # Number of critic networks.
    n_critics: int = 2
    # Number of flow denoising steps.
    denoise_steps: int = 10
    # Flow-matching training time distribution.
    t_sampling: Literal["grid", "uniform"] = "grid"
    # Action sampling and critic guidance method.
    sampling_mode: Literal["guided", "grad_step", "best_of_n", "bptt", "robust_q"] = "guided"
    # Critic guidance weight.
    guidance_weight: float = 1.0
    # Approximation used for guided denoising.
    denoised_action_approx: Literal["one_euler_step_approx", "noisy"] = "one_euler_step_approx"
    # Action gradient ascent step size.
    qgrad_step_size: float = 0.1
    # Number of action gradient ascent steps.
    qgrad_steps: int = 1
    # Use gradient signs for action gradient ascent.
    use_sign_gradient: bool = False
    # Number of action candidates for sampling.
    actor_num_samples: int = 32
    # Noise-conditioned robust critic learning rate.
    robust_critic_lr: float = 3e-4
    # Robust critic sinusoidal time embedding width.
    robust_critic_t_emb_size: int = 16
    # Hidden layer widths; None selects four layers of width 512.
    net_arch: Optional[tuple[int, ...]] = None
    # Apply layer normalization in the actor.
    actor_use_layer_norm: bool = True
    # Apply layer normalization in the critic.
    critic_use_layer_norm: bool = True
    # Apply layer normalization in the value network.
    value_use_layer_norm: bool = True
    # Network weight initialization.
    kernel_init: Optional[KernelInit] = "xavier_uniform"
    # Network backbone architecture.
    backbone_type: BackboneType = "mlp"
    # Network activation function.
    activation_fn: Optional[Activation] = "gelu"
