"""FQL-family learning parameters, independent of algorithms and training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import Activation, KernelInit
from rl_garden.networks.actor_critic import BackboneType

FacThreshold = Literal["batch_adaptive", "batch_wide_constant", "dataset_wide_constant"]
WeightType = Literal["linear", "logarithmic", "convex", "concave"]


@dataclass
class FQLParams(OfflineOptimizationArgs):
    """Optimization and network settings shared by all six learning cores."""

    # Target-network Polyak update coefficient.
    tau: float = 0.005

    # Actor learning rate.
    actor_lr: float = 0.0003

    # Critic learning rate.
    critic_lr: float = 0.0003

    # Euler integration steps for the action flow.
    flow_steps: int = 10

    # Critic ensemble aggregation for bootstrapping and action selection.
    q_agg: Literal['mean', 'min'] = 'mean'

    # Scale the actor Q objective by its absolute mean.
    normalize_q_loss: bool = False

    # Hidden layer widths; None selects four layers of width 512.
    net_arch: Optional[tuple[int, ...]] = None

    # Number of critic networks.
    n_critics: int = 2

    # Apply layer normalization to the actor networks.
    actor_use_layer_norm: bool = False

    # Apply layer normalization to the critic networks.
    critic_use_layer_norm: bool = True

    # Network weight initialization.
    kernel_init: Optional[KernelInit] = 'xavier_uniform'

    # Actor and scalar-critic backbone architecture.
    backbone_type: BackboneType = 'mlp'

    # Network activation function.
    activation_fn: Optional[Activation] = 'gelu'


@dataclass
class FQLCriticParams(FQLParams):
    """Settings consumed by the scalar critics, excluding ValueFlows."""

    # Apply group normalization to the scalar critics.
    critic_use_group_norm: bool = False

    # Number of scalar-critic normalization groups.
    num_groups: int = 32

    # Scalar-critic dropout probability, or None to disable.
    critic_dropout_rate: Optional[float] = None


@dataclass
class FQLDistillationParams(FQLParams):
    """Actor distillation settings used by every core except FAC."""

    # Actor flow-distillation objective weight.
    alpha: float = 10.0


@dataclass
class FQLReferenceParams(FQLCriticParams, FQLDistillationParams):
    """FQL learning settings and the published state-MLP execution recipe."""

    # Use the published FQL state-MLP networks and update recipe.
    reference_compatible: bool = False


@dataclass
class FloQParams(FQLCriticParams, FQLDistillationParams):
    """Learning settings consumed by FloQ."""

    # Minimum expected reward for flow-critic support.
    r_min: float = -1.0

    # Maximum expected reward for flow-critic support.
    r_max: float = 0.0

    # Number of flow critics.
    flow_num_ensembles: int = 2

    # Noise samples per transition for flow-critic regression.
    noise_samples: int = 8

    # Fraction of return support covered by initial flow noise.
    noise_coverage: float = 0.1

    # Euler steps for integrating the flow critic.
    critic_flow_steps: int = 8

    # Regress the flow critic only at zero flow time.
    train_at_zero_only: bool = False

    # Embed flow time in the critic velocity network.
    embed_time: bool = True

    # Width of the flow-time embedding.
    time_embed_dim: int = 64

    # Embed scalar returns using probability bins.
    use_prob_embed: bool = True

    # Number of return-probability bins.
    num_bins: int = 51

    # Width of the probability embedding.
    sigma: float = 16.0

    # Reward margin used to size the return support.
    reward_offset: float = 0.01

    # Flow-critic widths; None uses the resolved actor architecture.
    critic_flow_net_arch: Optional[tuple[int, ...]] = None


@dataclass
class FINOParams(FQLCriticParams, FQLDistillationParams):
    """Learning settings consumed by FINO."""

    # Noise injected into the BC-flow input during training.
    noise_scale: float = 0.1

    # Boltzmann action-selection scale; the policy owns its live value.
    beta: float = 10.0

    # Number of action candidates; FINO resolves None from the action dimension.
    num_samples: Optional[int] = None


@dataclass
class FACParams(FQLCriticParams):
    """Learning settings consumed by FAC."""

    # Conservative critic penalty weight.
    fac_alpha: float = 1.0

    # FAC actor-distillation objective weight.
    fac_lambda: float = 1.0

    # Behavior-density baseline used to gate the conservative penalty.
    fac_threshold: FacThreshold = 'batch_adaptive'

    # Flow log-density divergence estimator.
    logp_method: Literal["exact", "hutch-rade", "hutch-gaus"] = 'exact'

    # Number of Hutchinson divergence probes.
    logp_hutch_probes: int = 8

    # Mapping from density difference to conservative penalty weight.
    weight_type: WeightType = 'linear'

    # Behavior-flow learning rate.
    bc_lr: float = 0.0003

    # BC-prelude batch size; None uses the dataset-size heuristic.
    bc_batch_size: Optional[int] = None

    # BC-prelude epochs when no explicit update budget is supplied.
    bc_pretrain_epochs: int = 250

    # BC-prelude updates; None resolves from epochs and dataset size.
    bc_pretrain_steps: Optional[int] = None


@dataclass
class ValueFlowsParams(FQLDistillationParams):
    """Learning settings consumed by ValueFlows."""

    # Minimum reward for clipping flow returns.
    min_reward: float = -1.0

    # Maximum reward for clipping flow returns.
    max_reward: float = 0.0

    # Flow-critic aggregation for return targets.
    ret_agg: Literal['mean', 'min'] = 'mean'

    # Temperature of the critic confidence weights.
    confidence_weight_temp: float = 0.3

    # Differential consistency flow-matching loss weight.
    dcfm_lambda: float = 1.0

    # Bellman consistency flow-matching loss weight.
    bcfm_lambda: float = 1.0

    # Clip integrated returns to the reward-derived support.
    clip_flow_returns: bool = True

    # Number of action candidates for rejection sampling.
    num_samples: int = 16

    # Action-selection rule; online fine-tuning switches to rpg.
    policy_extraction: Literal['rs', 'rpg'] = 'rs'


@dataclass
class ACFQLParams(FQLCriticParams, FQLDistillationParams):
    """Learning settings consumed by ACFQL."""

    # Distilled one-step actor or best-of-n flow action selection.
    actor_type: Literal["distill-ddpg", "best-of-n"] = "distill-ddpg"

    # Number of candidate actions for best-of-n selection.
    actor_num_samples: int = 32

    # Actor flow-distillation objective weight.
    alpha: float = 100.0
