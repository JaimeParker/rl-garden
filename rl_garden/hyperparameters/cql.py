"""Learning settings for CQL and its calibrated offline-to-online branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.networks import KernelInit


@dataclass
class CQLParams(OfflineOptimizationArgs):
    """Shared constructor defaults for the CQL loss core."""

    tau: float = 0.005
    policy_lr: float = 0.0001
    q_lr: float = 0.0003
    alpha_lr: float = 0.0001
    cql_alpha_lr: float = 0.0003
    policy_frequency: int = 1
    target_network_frequency: int = 1
    ent_coef: float | str = 'auto'
    target_entropy: float | str = 'auto'
    backup_entropy: bool = False
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = None
    actor_use_layer_norm: bool = True
    critic_use_layer_norm: bool = True
    actor_use_group_norm: bool = False
    critic_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None
    kernel_init: Optional[KernelInit] = None
    backbone_type: Literal['mlp', 'mlp_resnet'] = 'mlp'
    std_parameterization: Literal['exp', 'uniform'] = 'exp'
    policy_log_std_multiplier: Optional[float] = None
    policy_log_std_offset: Optional[float] = None
    n_critics: int = 10
    critic_subsample_size: Optional[int] = 2
    use_cql_loss: bool = True
    cql_n_actions: int = 10
    cql_alpha: float = 5.0
    cql_autotune_alpha: bool = False
    cql_alpha_lagrange_init: float = 1.0
    cql_target_action_gap: float = 1.0
    cql_importance_sample: bool = True
    cql_max_target_backup: bool = True
    cql_temp: float = 1.0
    cql_clip_diff_min: float = float("-inf")
    cql_clip_diff_max: float = float("inf")
    cql_action_sample_method: Literal['uniform', 'normal'] = 'uniform'
    cql_penalty_scale: Literal['lagrange_only', 'lagrange_times_alpha'] = 'lagrange_only'
    cql_diff_clip_mode: Literal['skip_when_autotune', 'always'] = 'skip_when_autotune'
    cql_alpha_param: Literal['softplus', 'exp_clip'] = 'softplus'
    use_td_loss: bool = True


@dataclass
class CalQLParams(CQLParams):
    """Calibrated conservative Q learning and reference values."""

    use_calql: bool = True
    calql_bound_random_actions: bool = False
    sparse_reward_mc: bool = False
    sparse_negative_reward: float = 0.0
    success_threshold: float = 0.5
    use_sarsa_reference: bool = False
    sarsa_hidden_dims: tuple[int, ...] = (256, 256)
    sarsa_lr: float = 0.0003


@dataclass
class WSRLParams(CalQLParams):
    """Calibrated rollout settings, with conservatism disabled online."""

    online_cql_alpha: float = 0.0
    online_use_cql_loss: bool = False


@dataclass
class Off2OnCalQLParams(WSRLParams):
    """Keep the conservative objective during online fine-tuning."""

    online_cql_alpha: float = 5.0
    online_use_cql_loss: bool = True
