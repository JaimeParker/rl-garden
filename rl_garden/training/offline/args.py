"""Composable CLI arguments for offline algorithms."""
from __future__ import annotations

from rl_garden.hyperparameters.offline_sac import EDACParams

from dataclasses import dataclass
from typing import Literal, Optional

from rl_garden.hyperparameters.dreamer import RSSMPretrainParams
from rl_garden.hyperparameters.tdmpc2 import TDMPC2MultitaskParams
from rl_garden.hyperparameters.pwm import PWMParams
from rl_garden.common.cli_args import CheckpointArgs, LoggingArgs, ObservationArgs
from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.hyperparameters.fql import (
    FQLReferenceParams, FloQParams, FINOParams, FACParams, ValueFlowsParams,
)
from rl_garden.hyperparameters.qam import QAMParams
from rl_garden.hyperparameters.qgf import QGFParams
from rl_garden.hyperparameters.bcq import BCQParams
from rl_garden.hyperparameters.plas import PLASParams
from rl_garden.hyperparameters.idql import IDQLParams
from rl_garden.hyperparameters.awac import AWACParams
from rl_garden.hyperparameters.iql import IQLParams
from rl_garden.hyperparameters.td3_bc import TD3BCParams
from rl_garden.hyperparameters.rebrac import ReBRACParams
from rl_garden.hyperparameters.spot import SPOTParams
from rl_garden.hyperparameters.bc import BCParams
from rl_garden.hyperparameters.flow_bc import FlowBCParams
from rl_garden.hyperparameters.mean_flow_bc import MeanFlowBCParams
from rl_garden.hyperparameters.diffusion_bc import DiffusionBCParams
from rl_garden.hyperparameters.consistency_distill_bc import ConsistencyDistillBCParams
from rl_garden.hyperparameters.a2a_bc import A2ABCParams
from rl_garden.hyperparameters.bppo import BPPOParams
from rl_garden.hyperparameters.unio4 import UniO4Params
from rl_garden.hyperparameters.unio4_ope import UniO4OPEParams
from rl_garden.hyperparameters.hilp import HILPParams
from rl_garden.hyperparameters.opal import OPALParams
from rl_garden.hyperparameters.cql import CQLParams, CalQLParams, WSRLParams
from rl_garden.hyperparameters import shared
from rl_garden.hyperparameters.shared import (
    OfflineCriticArgs as OfflineCriticArgs,
    OfflineActorArgs as OfflineActorArgs,
    OfflineReplayArgs as OfflineReplayArgs,
    OfflineDiscountArgs as OfflineDiscountArgs,
    OfflineOptimizationArgs as OfflineOptimizationArgs,
)


OfflineDeterministicActorCriticArgs = shared.OfflineDeterministicActorCriticArgs


@dataclass
class OfflineDatasetArgs:
    num_offline_steps: int = 100_000
    # Registry-backed (rl_garden.buffers.dataset_backend_registry), not a
    # Literal: a new backend registers itself, no CLI arg change needed here.
    dataset_backend: str = "h5"

    # The dataset locator is a filesystem path for "h5", a dataset id for
    # "minari", and a legacy Gym environment id for "d4rl_legacy".
    offline_dataset: Optional[str] = None

    offline_num_traj: Optional[int] = None
    save_filename: Optional[str] = None
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    success_key: Optional[str] = None
    action_low: float = -1.0
    action_high: float = 1.0
    spec_num_envs: int = 1


@dataclass
class OfflineRuntimeArgs:
    seed: int = 1


@dataclass
class OfflineEvalArgs:
    env_id: Optional[str] = None
    num_eval_envs: int = 1
    num_eval_episodes: int = 100
    # Vector-env steps per evaluation (whole chunks with action chunking).
    # None derives num_eval_episodes * eval_episode_horizon when the horizon
    # is set.
    num_eval_steps: Optional[int] = None

    # Expected worst-case env steps for one eval episode (e.g. 1000 for
    # AntMaze). Only sizes the eval step budget when num_eval_steps is unset
    # (num_eval_steps = num_eval_episodes * eval_episode_horizon) -- does not
    # impose a TimeLimit on the eval env itself.
    eval_episode_horizon: Optional[int] = None

    control_mode: str = "pd_joint_delta_pos"
    render_mode: str = "rgb_array"


@dataclass
class OfflineCommonArgs(
    OfflineEvalArgs,
    OfflineRuntimeArgs,
    ObservationArgs,
    OfflineOptimizationArgs,
    OfflineReplayArgs,
    OfflineDatasetArgs,
    LoggingArgs,
    CheckpointArgs,
    EnvBackendArgs,
):
    """Arguments shared by every offline algorithm."""


@dataclass
class TDMPC2MultitaskTrainingArgs(TDMPC2MultitaskParams, OfflineRuntimeArgs, CheckpointArgs, LoggingArgs):
    """TD-MPC2 multitask offline pretraining.

    Deliberately does NOT inherit ``EnvRunArgs``/``EnvBackendArgs``/
    ``OfflineDatasetArgs``: there is no single ``env_id``/live env (training
    never touches one, see ``rl_garden.algorithms.tdmpc2_multitask``)
    and no single homogeneous dataset (``dataset_dir`` points at the
    per-task, differently-shaped output of
    ``tools/conversion/convert_tdmpc2_multitask_dataset.py``, not one
    ``offline_dataset`` file).
    """

    dataset_dir: str = ""
    mmap_dir: str = ""
    # Load only these manifest tasks (empty = all) and each task's first N
    # episodes (0 = all); task ids keep their full-manifest indices.
    tasks: tuple[str, ...] = ()
    max_episodes_per_task: int = 0

    device: str = "auto"
    num_offline_steps: int = 10_000_000
    buffer_size: int = 1_000_000
    batch_size: int = 256


@dataclass
class PWMTrainingArgs(PWMParams, OfflineRuntimeArgs, CheckpointArgs, LoggingArgs):
    """PWM offline policy learning on a frozen multitask world model.

    Same base set as ``TDMPC2MultitaskTrainingArgs`` (no ``EnvRunArgs``/
    ``OfflineDatasetArgs``): ``dataset_dir`` is the converted TD-MPC2
    multitask dataset, ``task`` picks one of its tasks and doubles as the
    eval ``env_id`` on ``eval_env_backend``. Evaluation runs ``eval_runs``
    episodes in parallel; its step cap is derived from the task's episode
    length (``num_eval_steps`` is not read).
    """

    dataset_dir: str = ""
    mmap_dir: str = ""
    world_model_checkpoint: str = ""
    task: str = ""
    eval_env_backend: str = "dm_control"
    device: str = "auto"
    num_offline_steps: int = 10_000
    batch_size: int = 512
    gamma: float = 0.99
    # Evaluate every this many gradient updates (0 disables).
    eval_freq: int = 200

    eval_runs: int = 10


@dataclass
class RSSMPretrainTrainingArgs(RSSMPretrainParams, OfflineRuntimeArgs, CheckpointArgs, LoggingArgs):
    """Reward-free RSSM world-model pretraining (NCRL stage 1, see
    ``rl_garden.algorithms.rssm_pretrain``). Same base set as
    ``TDMPC2MultitaskTrainingArgs``: no live env and no ``offline_dataset``.
    ``dataset_dir`` is the output of ``tools/conversion/convert_ncrl_dataset.py``
    (one embodiment directory, or a root of several that are mixed by
    episode). ``size`` names a ``RSSM_SIZES`` row; ``deter``/``hidden``/
    ``classes``/``cnn_depth`` override single fields of it.
    ``padded_action_dim=None`` takes the manifest's value. ``backbone="ncrl"``
    (default) builds the official NCRL networks (64x64 images only);
    ``"dreamer_v3"`` the DreamerV3 ones."""

    dataset_dir: str = ""
    dataset_backend: Literal["mmap", "buffer"] = "mmap"
    backbone: Literal["dreamer_v3", "ncrl"] = "ncrl"
    device: str = "auto"
    buffer_device: str = "cuda"
    num_offline_steps: int = 200_000
    batch_size: int = 32
    compute_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    padded_action_dim: Optional[int] = None


@dataclass
class DiffusionBCTrainingArgs(DiffusionBCParams, ObservationArgs, CheckpointArgs, LoggingArgs):
    """Diffusion BC pretraining (DPPO phase 1). Deliberately does NOT inherit
    ``OfflineCommonArgs``: ``run_offline`` assumes a ``agent.replay_buffer``
    populated via ``load_offline_dataset``, but ``DiffusionBC`` loads
    ``(obs_history, action_chunk)`` windows directly in its constructor (see
    ``rl_garden.buffers.chunked_dataset``) and has no replay buffer at all --
    same reasoning as ``TDMPC2MultitaskTrainingArgs``. Adds ``ObservationArgs``
    (absorbed from the former standalone ``VisionDiffusionBCTrainingArgs``) so
    a single ``diffusion_bc`` CLI surface covers both Box and Dict (vision)
    observation spaces, matching ``DiffusionBC``'s in-class
    ``isinstance(obs_space, spaces.Box/Dict)`` branch -- most ``encoder``/
    ``obs_groups`` fields are no-ops for Box-obs (state-only) datasets."""
    dataset_path: str = ""
    num_offline_steps: int = 200_000
    offline_num_traj: Optional[int] = None
    net_backbone: Literal["mlp", "unet"] = "mlp"
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    unet_cond_predict_scale: bool = False
    batch_size: int = 128
    seed: int = 1
    device: str = "auto"


@dataclass
class ConsistencyDistillBCTrainingArgs(ConsistencyDistillBCParams, CheckpointArgs, LoggingArgs):
    """Fully offline LCM-style consistency distillation of a frozen
    ``DiffusionBC`` teacher into a one/few-step ``ConsistencyDistillBC``
    student. Same "no ``OfflineCommonArgs``, no replay buffer" reasoning as
    ``DiffusionBCTrainingArgs`` -- dataset is loaded directly, bespoke runner.

    ``horizon_steps``/``cond_steps``/``denoising_steps``/``net_backbone``+
    ``unet_*``/``time_dim``/``kernel_init`` must match whatever the
    ``--bc_checkpoint`` teacher was trained with: student/target networks are
    warm-started from its EMA state dict.
    ``ConsistencyDistillBC._setup_model`` validates the teacher-matching
    fields against the checkpoint's own recorded hyperparameters before
    loading and raises ``ValueError`` naming every mismatch (a `net_cls`
    mismatch alone would also fail via a state-dict shape error, but fields
    like ``denoising_steps``/``activation_fn`` carry no parameters and would
    otherwise load silently wrong)."""
    dataset_path: str = ""
    bc_checkpoint: str = ""
    num_offline_steps: int = 100_000
    offline_num_traj: Optional[int] = None
    net_backbone: Literal["mlp", "unet"] = "mlp"
    unet_down_dims: tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8
    unet_cond_predict_scale: bool = False
    batch_size: int = 128
    seed: int = 1
    device: str = "auto"


@dataclass
class HILPTrainingArgs(HILPParams, CheckpointArgs, LoggingArgs):
    """HILP pretraining. Deliberately does NOT inherit ``OfflineCommonArgs``:
    ``run_offline`` assumes an ``agent.replay_buffer`` populated via
    ``load_offline_dataset``, but ``HILP`` loads a ``HindsightGoalDataset``
    directly in its constructor and has no replay buffer at all -- same
    reasoning as ``DiffusionBCTrainingArgs``/``A2ABCTrainingArgs``. State-based
    (Box observations) only."""

    dataset_path: str = ""
    num_offline_steps: int = 1_000_000
    offline_num_traj: Optional[int] = None

    batch_size: int = 1024
    seed: int = 1
    device: str = "auto"


@dataclass
class OPALTrainingArgs(OPALParams, CheckpointArgs, LoggingArgs):
    """OPAL VAE pretraining. Deliberately does NOT inherit ``OfflineCommonArgs``:
    ``run_offline`` assumes an ``agent.replay_buffer`` populated via
    ``load_offline_dataset``, but ``OPAL`` loads a ``WindowedTrajectoryDataset``
    directly in its constructor and has no replay buffer at all -- same
    reasoning as ``HILPTrainingArgs``/``A2ABCTrainingArgs``. State-based
    (Box observations) only."""

    dataset_path: str = ""
    num_offline_steps: int = 1_000_000
    offline_num_traj: Optional[int] = None

    batch_size: int = 256
    seed: int = 1
    device: str = "auto"


@dataclass
class A2ABCTrainingArgs(A2ABCParams, ObservationArgs, CheckpointArgs, LoggingArgs):
    """A2A flow-matching BC pretraining. A standalone sibling of
    ``DiffusionBCTrainingArgs`` (not built on it) -- swaps
    diffusion-specific fields (``denoising_steps``, ``ema_*``,
    ``residual_style``, ``time_dim``) for A2A's flow-in-latent-space fields.
    ``obs.state`` must stay True (``A2ABC._setup_model`` raises ``ValueError``
    otherwise) -- the state-history window is the flow's source, not
    optional."""
    dataset_path: str = ""
    num_offline_steps: int = 200_000
    offline_num_traj: Optional[int] = None
    activation_fn: Literal["relu", "gelu", "mish"] = "relu"
    kernel_init: Optional[
            Literal["xavier_uniform", "xavier_normal", "orthogonal", "kaiming_uniform"]
        ] = None
    batch_size: int = 128
    seed: int = 1
    device: str = "auto"


@dataclass
class OfflineDeviceArgs:
    device: str = "auto"


@dataclass
class OfflineSACNetworkArgs:
    hidden_dim: int = 256
    actor_hidden_layers: int = 2
    critic_hidden_layers: int = 4


@dataclass
class OfflineValueArgs:
    value_use_layer_norm: bool = False
    value_use_group_norm: bool = False
    value_dropout_rate: Optional[float] = None


@dataclass
class OfflineCompileArgs:
    use_compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


@dataclass
class OfflineCQLArgs(CQLParams):
    """Learning settings for the OfflineCQLArgs entry family."""




@dataclass
class OfflineCalQLArgs(CalQLParams):
    """Learning settings for the OfflineCalQLArgs entry family."""




@dataclass
class OfflineIQLArgs(IQLParams):
    """IQL hyperparameters for offline and offline-to-online training."""


@dataclass
class OfflineIDQLArgs(IDQLParams):
    """IDQL parameters for offline training."""


@dataclass
class OfflineBCArgs(BCParams):
    """BC policy-update parameters."""



@dataclass
class OfflineFlowBCArgs(FlowBCParams):
    """FlowBC-specific network/training knobs. Deliberately not built on
    ``OfflineActorArgs`` -- that class's ``backbone_type``/``std_parameterization``/
    ``actor_use_group_norm``/``num_groups``/``actor_dropout_rate`` fields are
    Gaussian-actor-specific and have no ``ActorVectorField`` equivalent; reusing
    it would expose CLI flags that silently do nothing."""


@dataclass
class OfflineMeanFlowBCArgs(MeanFlowBCParams):
    """MeanFlowBC-specific network/training knobs. Same rationale as
    ``OfflineFlowBCArgs`` for not building on ``OfflineActorArgs``."""


@dataclass
class OfflineWSRLArgs(WSRLParams):
    """Learning settings for the OfflineWSRLArgs entry family."""

    training_freq: int = 64



@dataclass
class OfflineTD3BCArgs(TD3BCParams):
    """TD3BC hyperparameters for offline training."""


@dataclass
class OfflineReBRACArgs(ReBRACParams):
    """ReBRAC hyperparameters for offline training."""


@dataclass
class OfflineAWACArgs(AWACParams):
    """AWAC hyperparameters for offline training."""


@dataclass
class OfflineFQLArgs(FQLReferenceParams):
    """FQL parameters for offline training."""

    actor_use_group_norm: bool = False

    actor_dropout_rate: Optional[float] = None
    # encoder_sharing lives on ObservationArgs (OfflineCommonArgs already
    # mixes it in): "shared_critic_grad" (default, AGENTS.md's project
    # convention, matches SACPolicy) vs. "separate" (three independent
    # encoder instances, matching FQL's own JAX reference). Only meaningful
    # for Dict (vision) observation spaces -- Box observations use a
    # parameterless FlattenExtractor either way.


@dataclass
class OfflineFloQArgs(FloQParams):
    """FloQ parameters for offline training."""

    actor_use_group_norm: bool = False

    actor_dropout_rate: Optional[float] = None


@dataclass
class OfflineValueFlowsArgs(ValueFlowsParams):
    """ValueFlows parameters for offline training."""

    actor_use_group_norm: bool = False

    actor_dropout_rate: Optional[float] = None

    critic_use_group_norm: bool = False

    num_groups: int = 32

    critic_dropout_rate: Optional[float] = None


@dataclass
class OfflineFINOArgs(FINOParams):
    """FINO parameters for offline training."""

    actor_use_group_norm: bool = False

    actor_dropout_rate: Optional[float] = None


@dataclass
class OfflineFACArgs(FACParams):
    """FAC parameters for offline training."""

    actor_use_group_norm: bool = False

    actor_dropout_rate: Optional[float] = None

    alpha: float = 10.0


@dataclass
class OfflineQGFArgs(QGFParams):
    """QGF parameters for offline training."""

    horizon_length: int = 1
    actor_use_group_norm: bool = False
    critic_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None


@dataclass
class OfflineQAMArgs(QAMParams):
    """QAM parameters for offline training."""

    horizon_length: int = 1
    actor_use_group_norm: bool = False
    critic_use_group_norm: bool = False
    num_groups: int = 32
    actor_dropout_rate: Optional[float] = None
    critic_dropout_rate: Optional[float] = None


@dataclass
class OfflineEDACArgs(EDACParams):
    """EDAC hyperparameters. Defaults match CORL's ``edac.py::TrainConfig``.
    Field names follow ``OfflineSAC``'s own convention (``policy_lr``/
    ``q_lr``/``alpha_lr``, not TD3-BC-family's ``actor_lr``/``critic_lr``),
    since ``EDAC`` subclasses ``OfflineSAC`` directly."""

    alpha_lr: Optional[float] = 3e-4
    # CORL's Actor/VectorizedCritic both use 3 hidden layers of width 256.
    net_arch: Optional[tuple[int, ...] | dict[str, tuple[int, ...]]] = (256, 256, 256)


@dataclass
class OfflineBCQArgs(BCQParams):
    """BCQ parameters for offline training."""


@dataclass
class OfflinePLASArgs(PLASParams):
    """PLAS parameters for offline training."""


@dataclass
class OfflineSPOTArgs(SPOTParams):
    """SPOT hyperparameters for offline training."""


@dataclass
class OfflineBPPOArgs(BPPOParams):
    """BPPO hyperparameters. Defaults match ``3rd_party/BPPO/main.py``.

    ``critic_warmup_steps`` (Phase A, V/Q via MC-return/SARSA) and the actor
    phase (Phase B, PPO-clip) share one ``--num_offline_steps`` budget --
    ``run_bppo`` asserts ``num_offline_steps > critic_warmup_steps`` so the
    actor phase is never silently skipped. ``bc_checkpoint`` (optional) warm-
    starts the actor from a ``BC`` pretrained checkpoint via
    ``BPPO.load_actor_from`` -- not the generic ``--load_checkpoint`` path,
    which would pollute BPPO's own step counters with BC's unrelated
    training length.
    """

    bc_checkpoint: Optional[str] = None


@dataclass
class OfflineUniO4Args(UniO4Params):
    """Uni-O4 hyperparameters. Defaults match ``3rd_party/Uni-O4/main.py``
    (a separate default config from BPPO's own script -- note
    ``actor_hidden_dims``/``omega`` differ from ``OfflineBPPOArgs``'s
    defaults on purpose).

    Shares one critic across the ensemble (see ``rl_garden/algorithms
    /unio4.py``'s module docstring), so there is no per-member value/Q
    config here -- ``critic_warmup_steps``/``value_lr``/etc. below configure
    that one shared critic, identical in shape to ``OfflineBPPOArgs``'s.
    """


@dataclass
class OfflineUniO4OPEArgs(UniO4OPEParams):
    """UniO4OPE: Uni-O4 with dynamics-model OPE gating (Milestone A, see
    ``rl_garden/algorithms/unio4_ope.py``). Dynamics hyperparameters default
    to ``3rd_party/Uni-O4/transition_model/configs/gym/default.py``'s values
    (shared across halfcheetah/hopper/walker2d -- task configs there only
    ever override ``rollout_length``/``penalty_coef``, both dead for this
    call path, see ``unio4_ope.py``'s module docstring)."""
