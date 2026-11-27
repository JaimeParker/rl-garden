"""CLI argument dataclasses for offline-to-online training.

``Off2OnCommonArgs`` holds orchestration/training fields generic across any
off2on algorithm family (Cal-QL and IQL today). ``CQLOff2OnArgs`` holds
CQL/Cal-QL-only hyperparameters. ``WSRLTrainingArgs`` composes both (kept as
one name/field-set for backward compat with ``off2on/entrypoints/wsrl.py``/
``off2on/entrypoints/calql.py``, which reference it unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import warnings

from rl_garden.common.cli_args import CheckpointArgs, ObservationArgs
from rl_garden.common.env_args import EnvRunArgs
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.training.offline.args import OfflineIQLArgs
from rl_garden.hyperparameters.awac import AWACParams
from rl_garden.hyperparameters.cql import WSRLParams
from rl_garden.hyperparameters.spot import Off2OnSPOTParams
from rl_garden.hyperparameters.shared import (
    Off2OnLoopArgs as Off2OnLoopArgs,
    OfflineActorArgs,
    OfflineCriticArgs,
    OfflineOptimizationArgs,
)


@dataclass
class Off2OnRunArgs(EnvRunArgs, CheckpointArgs):
    """Run settings of an offline-to-online run."""

    # Vector-env steps per evaluation (whole chunks with action chunking).
    # Replaces LoggingArgs's fixed int=50 default with the same "unset"
    # sentinel offline training uses, so resolve_num_eval_steps can derive a
    # budget from eval_episode_horizon below when appropriate.
    num_eval_steps: Optional[int] = None
    # Expected worst-case env steps for one eval episode (e.g. 1000 for
    # AntMaze). Only sizes the eval step budget when num_eval_steps is unset
    # and the algorithm's own num_eval_episodes field is set -- does not
    # impose a TimeLimit on the eval env itself.
    eval_episode_horizon: Optional[int] = None

    num_offline_steps: int = 0
    num_online_steps: int = 1_000_000
    # Registry-backed (rl_garden.buffers.dataset_backend_registry), not a
    # Literal: a new backend registers itself, no CLI arg change needed here.
    dataset_backend: str = "h5"

    # The dataset locator is a filesystem path for "h5", a dataset id for
    # "minari", and a legacy Gym environment id for "d4rl_legacy".
    offline_dataset: Optional[str] = None

    offline_num_traj: Optional[int] = None
    action_low: float = -1.0
    action_high: float = 1.0
    online_replay_mode: Literal["empty", "append", "mixed"] = "empty"
    offline_data_ratio: float | str = 0.0
    warmup_steps: int = 5000
    success_key: Optional[str] = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Phase-specific eval cadences. Offline uses gradient-update units in the
    # manual offline loop; online uses env-step units through OffPolicyAlgorithm.
    offline_eval_freq: Optional[int] = None
    online_eval_freq: Optional[int] = None


@dataclass
class Off2OnCommonArgs(
    Off2OnRunArgs,
    Off2OnLoopArgs,
    OfflineOptimizationArgs,
    OfflineCriticArgs,
    OfflineActorArgs,
):
    """Arguments shared by every offline-to-online entry."""


@dataclass
class FQLTrainingArgs(Off2OnRunArgs, Off2OnLoopArgs):
    """FQL-family orchestration and compatibility flags."""

    actor_use_group_norm: bool = False
    actor_dropout_rate: Optional[float] = None
    critic_subsample_size: int = 2
    std_parameterization: Literal["exp", "uniform"] = "exp"


@dataclass
class ValueFlowsTrainingArgs(FQLTrainingArgs):
    """Compatibility options for ValueFlows' replaced scalar critic."""

    critic_use_group_norm: bool = False
    num_groups: int = 32
    critic_dropout_rate: Optional[float] = None


@dataclass
class CQLOff2OnArgs(WSRLParams):
    """CQL/Cal-QL rollout learning and compilation settings."""

    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"



@dataclass
class WSRLTrainingArgs(Off2OnRunArgs, CQLOff2OnArgs):
    """CQL/Cal-QL rollout orchestration settings."""

    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    offline_sampling: Literal["with_replace", "without_replace"] = "with_replace"
    gamma: float = 0.99
    utd: float = 4.0
    learning_starts: int = 4_000
    training_freq: int = 64



@dataclass
class VisionWSRLTrainingArgs(WSRLTrainingArgs, ObservationArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class IQLOff2OnTrainingArgs(Off2OnCommonArgs, OfflineIQLArgs):
    """IQL off2on args: ``Off2OnCommonArgs`` + IQL-specific hyperparameters
    (reused from ``rl_garden.training.offline.args``)."""


@dataclass
class VisionIQLOff2OnTrainingArgs(IQLOff2OnTrainingArgs, ObservationArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class AWACOff2OnHyperparamArgs(AWACParams):
    """AWAC hyperparameters for offline-to-online training."""


@dataclass
class AWACOff2OnTrainingArgs(AWACOff2OnHyperparamArgs, Off2OnCommonArgs):
    """AWAC off2on args: ``Off2OnCommonArgs`` + AWAC-specific hyperparameters.

    AWAC is Box-observation only: no ``ObservationArgs`` mixed in, so it has
    no ``--obs``/``--encoder`` CLI surface at all.
    """


@dataclass
class SPOTOff2OnHyperparamArgs(Off2OnSPOTParams):
    """SPOT hyperparameters for offline-to-online training."""


@dataclass
class SPOTOff2OnTrainingArgs(SPOTOff2OnHyperparamArgs, Off2OnCommonArgs):
    """SPOT off2on args: ``Off2OnCommonArgs`` + SPOT-specific hyperparameters.

    This class alone has no ``--obs``/``--encoder`` CLI surface;
    ``SPOTOff2OnArgs`` (``rl_garden/training/off2on/entrypoints/spot.py``) mixes in
    ``ObservationArgs`` separately, so SPOT does accept Box or Dict
    (vision) observations, ``obs_groups``, and ``critic_encoder``/
    ``encoder_sharing`` at the CLI.
    """


def initial_training_phase_from_args(
    args: Off2OnCommonArgs,
) -> Optional[InitialTrainingPhase]:
    if args.warmup_steps <= 0:
        return None
    return InitialTrainingPhase(
        duration_steps=args.warmup_steps,
        update_actor=False,
        update_critic=False,
        update_encoder=False,
        random_action_prob=0.0,
    )


def warn_if_off2on_warmup_uses_uninitialized_policy(args: Off2OnCommonArgs) -> None:
    if args.warmup_steps > 0 and args.load_checkpoint is None:
        warnings.warn(
            "warmup_steps > 0 without an offline-trained checkpoint will use "
            "the randomly initialized policy to collect data while updates are "
            "paused. Use --load_checkpoint or set --warmup_steps 0 unless this "
            "is intentional.",
            UserWarning,
            stacklevel=2,
        )
