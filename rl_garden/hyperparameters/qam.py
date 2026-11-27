"""QAM learning hyperparameters shared by constructors and training Args."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType


@dataclass
class QAMParams:
    """Defaults of the QAM learning algorithm."""

    # Target critic Polyak update coefficient.
    tau: float = 0.005
    # Actor learning rate.
    actor_lr: float = 3e-4
    # Critic learning rate.
    critic_lr: float = 3e-4
    # Optimizer weight decay.
    weight_decay: float = 0.0
    # Use decoupled AdamW weight decay.
    use_adamw: bool = False
    # Learning-rate schedule.
    lr_schedule: Literal["constant", "linear_warmup", "warmup_cosine"] = "constant"
    # Number of learning-rate warmup updates.
    lr_warmup_steps: int = 0
    # Number of learning-rate decay updates.
    lr_decay_steps: int = 0
    # Minimum learning-rate fraction.
    lr_min_ratio: float = 0.0
    # Maximum gradient norm, or None to disable clipping.
    grad_clip_norm: Optional[float] = 1.0
    # Critic bootstrap objective.
    critic_loss_type: Literal["ddpg", "iql"] = "ddpg"
    # Critic ensemble disagreement penalty.
    rho: float = 0.0
    # Expectile of the value regression.
    expectile: float = 0.9
    # Number of critic networks.
    n_critics: int = 2
    # Number of flow integration steps.
    flow_steps: int = 10
    # Number of candidate actions for best-of-N inference.
    best_of_n: int = 1
    # Inverse temperature of critic guidance.
    inv_temp: float = 0.3
    # Represent the fast flow as a residual over the slow flow.
    residual: bool = False
    # Use the target slow actor for adjoint matching.
    target_actor: bool = True
    # Clip adjoint guidance.
    clip_adj: bool = True
    # Use target critic gradients for adjoint matching.
    use_target_grad: bool = True
    # One-step flow distillation weight; mutually exclusive with edit_scale.
    fql_alpha: float = 0.0
    # Residual edit action scale; mutually exclusive with fql_alpha.
    edit_scale: float = 0.0
    # Edit policy entropy target; None derives it from the action dimension.
    edit_target_entropy: Optional[float] = None
    # Multiplier for the derived edit entropy target.
    edit_target_entropy_multiplier: float = 0.5
    # Edit entropy coefficient learning rate.
    edit_alpha_lr: float = 3e-4
    # Hidden layer widths; None selects four layers of width 512.
    net_arch: Optional[tuple[int, ...]] = None
    # Apply layer normalization in the actor.
    actor_use_layer_norm: bool = False
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
