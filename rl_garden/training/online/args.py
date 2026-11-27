"""CLI argument dataclasses for online training algorithms."""
from __future__ import annotations

from rl_garden.hyperparameters.flash_sac import FlashSACParams

from rl_garden.hyperparameters.dreamer import DreamerV3Params, NCRLParams
from rl_garden.hyperparameters.sac import SACParams, RecurrentSACParams, TransformerSACParams, SACFlowParams

from rl_garden.hyperparameters.tdmpc2 import TDMPC2Params
from rl_garden.hyperparameters.supermimic import SuperMimicParams
from rl_garden.hyperparameters.ppo import PPOParams, RecurrentPPOParams, TransformerPPOParams, GAILParams
from rl_garden.hyperparameters.chain_ppo import DPPOParams, FlowPPOParams, DiffusionCMDistillOnlineParams
from rl_garden.hyperparameters.policy_distillation import PolicyDistillationParams

from dataclasses import dataclass, field
from typing import Literal, Optional

from rl_garden.common.cli_args import (
    CheckpointArgs,
    LoggingArgs,
    ObservationArgs,
    PolicyModuleArgs,
)
from rl_garden.common.env_args import EnvRunArgs
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.hyperparameters.dagger import DAggerParams
from rl_garden.observations.config import ObservationConfig


@dataclass
class SACTrainingArgs(SACParams, EnvRunArgs, CheckpointArgs):
    # Opt-in heterogeneous critic MLP backbone (asymmetric/privileged critic).
    # Only consulted by the SAC-family entrypoints that support a distinct
    # critic encoder (sac/rlpd/rlpd_hybrid); unset keeps today's exact
    # behavior (actor/critic share one backbone_type).
    critic_backbone_type: Optional[Literal["mlp", "mlp_resnet"]] = None

    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 1024
    learning_starts: int = 4_000
    training_freq: int = 64
    utd: float = 0.5
    gamma: float = 0.8
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"
    hidden_dim: int = 256
    actor_hidden_layers: int = 3
    critic_hidden_layers: int = 3
    q_landscape_diagnostics: bool = False
    q_landscape_num_actions: int = 8
    q_landscape_batch_size: int = 64
    q_mc_diagnostics: bool = False
    critic_only_steps: int = 0
    critic_only_freeze_encoder: bool = True
    critic_only_random_action_prob: float = 0.0
    load_actor_checkpoint: Optional[str] = None


@dataclass
class VisionSACTrainingArgs(SACTrainingArgs, ObservationArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25
    mmap_dir: Optional[str] = None
    mmap_mode: Literal["create", "open"] = "create"


@dataclass
class RecurrentSACTrainingArgs(RecurrentSACParams, SACTrainingArgs):
    pass


@dataclass
class VisionRecurrentSACTrainingArgs(RecurrentSACTrainingArgs, ObservationArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class TransformerSACTrainingArgs(TransformerSACParams, SACTrainingArgs):
    pass


@dataclass
class VisionTransformerSACTrainingArgs(TransformerSACTrainingArgs, ObservationArgs):
    buffer_size: int = 200_000
    batch_size: int = 512
    utd: float = 0.25


@dataclass
class SACFlowTrainingArgs(SACFlowParams, SACTrainingArgs):
    """SACFlow -- flow-matching actor. State observations (use
    ``VisionSACFlowTrainingArgs`` for Dict/RGBD)."""

    flow_hidden_dim: int = 256
    flow_hidden_layers: int = 3


@dataclass
class VisionSACFlowTrainingArgs(SACFlowTrainingArgs, ObservationArgs):
    """SACFlow, state-obs by default (``ObservationArgs.obs`` defaults to
    state-only) -- pass ``--obs.rgb <camera>`` to opt into Dict/RGBD
    observations (CNN-based encoders only; see ``SACFlow``'s own docstring
    for why ``--encoder.backbone vit`` and ``--critic-encoder`` are not
    supported this round). ``buffer_size``/``batch_size``/``utd`` stay at
    ``SACFlowTrainingArgs``'s existing (state-tuned) values -- a visual run
    should pass ``--buffer_size``/``--batch_size``/``--utd`` explicitly if
    the 1M-buffer state default isn't appropriate for image observations."""


@dataclass
class TDMPC2TrainingArgs(TDMPC2Params, EnvRunArgs, CheckpointArgs):
    """State-obs defaults, values matching
    ``3rd_party/tdmpc2/tdmpc2/config.yaml`` where an upstream default exists.

    ``num_envs``/``num_eval_envs`` are fixed to 1: TDMPC2's CEM planner already
    rolls out ``num_samples`` trajectories per env step, and vectorized
    rollout is not supported in this port (see ``rl_garden.algorithms.tdmpc2``
    module docstring). ``episode_length`` has no single upstream
    default (it's task-specific in the original Hydra configs); 100 matches
    the ManiSkill task horizon this port was originally tuned against.
    """

    num_envs: int = 1
    num_eval_envs: int = 1
    total_timesteps: int = 10_000_000
    episode_length: int = 100
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 256
    seed_steps: Optional[int] = None
    episodic: bool = False
    termination_coef: float = 1.0


@dataclass
class VisionTDMPC2TrainingArgs(TDMPC2TrainingArgs, ObservationArgs):
    buffer_size: int = 200_000


@dataclass
class DreamerV3TrainingArgs(DreamerV3Params, EnvRunArgs, CheckpointArgs, ObservationArgs):
    """DreamerV3 hyperparameters -- values match the r2dreamer/official-JAX defaults (see ``rl_garden.algorithms.dreamer_v3``
    module docstring); ``size12M`` preset default."""

    total_timesteps: int = 1_000_000
    horizon: float = 333.0
    # None means use --horizon.
    discount_horizon: Optional[float] = None
    batch_size: int = 16
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    learning_starts: int = 1_024
    # None (default) resolves to "bfloat16" on CUDA, "float32" on CPU --
    # see DreamerV3's own docstring.
    compute_dtype: Optional[Literal["float32", "bfloat16"]] = None


def _ncrl_default_observation() -> ObservationConfig:
    # The pretrained RSSM encoder is image-only (NCRL's 64x64 rgb).
    return ObservationConfig(state=False, rgb=("default",), image_size=(64, 64))


@dataclass
class NCRLTrainingArgs(NCRLParams, EnvRunArgs, CheckpointArgs, ObservationArgs):
    """NCRL stage-2 online fine-tuning (``rl_garden.algorithms.ncrl``).
    Defaults follow the official ``finetune.yaml`` + ``scripts/finetune.sh``
    (survey section 8). The RSSM architecture comes from
    ``--pretrained_world_model`` (an ``RSSMPretrain`` checkpoint), so the
    DreamerV3 size/layer flags do not exist here. ``episode_frames``/
    ``action_repeat`` default (``None``) to the dm_control backend's values
    (1000 frames, ``--dm_control.action_repeat``); other backends must set
    them. ``learning_starts=None`` means 10 episodes. Ablations:
    ``--online_ratio 1.0`` turns rehearsal off (and with it guidance, which
    trains on the retrieved data); ``--no-use_guidance`` turns guidance off.
    ``backbone`` must match the pretrained checkpoint's (checkpoints without
    one are ``dreamer_v3``)."""

    obs: ObservationConfig = field(default_factory=_ncrl_default_observation)
    total_timesteps: int = 250_000
    pretrained_world_model: str = ""
    backbone: Literal["dreamer_v3", "ncrl"] = "ncrl"
    retrieval_db: Optional[str] = None
    prior_dataset_dir: Optional[str] = None
    episode_frames: Optional[int] = None
    action_repeat: Optional[int] = None
    learning_starts: Optional[int] = None
    batch_size: int = 16
    padded_action_dim: Optional[int] = None
    # 250k policy steps = the official 500k-frame run. Each step stores a
    # uint8 64x64x3 frame (12 KiB) plus the float32 RSSM carry (ncrl_m:
    # (4096 + 32*48) * 4 B = 22 KiB), ~34 KiB/step -> ~8 GiB on buffer_device;
    # 500 prior episodes x 500 steps add another ~8 GiB.
    buffer_size: int = 250_000

    buffer_device: str = "cuda"
    compute_dtype: Optional[Literal["float32", "bfloat16"]] = None


@dataclass
class SuperMimicTrainingArgs(SuperMimicParams, EnvRunArgs, CheckpointArgs, ObservationArgs):
    """Defaults follow the official implementation (github.com/HybridRobotics/supermimic);
    the paper's Table A.5 lists a cosine schedule the official code does not use,
    so the schedule is off by default. SuperMimic backpropagates through imagined
    rollouts of an explicit-state world model.

    Privileged quantities travel as observation keys: declare them with
    ``--obs.extra_state sim ctx goal time`` and the env must emit
    ``state_sim`` (modeled simulator state), ``state_ctx`` (per-episode
    context), ``state_goal`` and ``state_time`` (reference time index). Each is
    optional; the default observation uses only ``state``. ``--heads analytic``
    needs the env to expose ``analytic_mdp`` and to emit ``state_time``. The
    state codec is a Python-API argument of ``SuperMimic`` (``codec``) and has
    no flag. Goal generation likewise accepts the Python object ``goal_source``
    or automatically uses ``env.unwrapped.goal_source``; there is no CLI flag.
    The world-model class (``world_model_cls``, e.g.
    ``NominalErrorStateSpaceWorldModel``) is also Python-API only.

    ``encoder``, ``obs_groups``, ``critic_encoder`` and ``encoder_sharing``
    (from ``ObservationArgs``) have no effect here: the policy trains no
    observation encoder, and the builder does not read them.

    The official schedule defines sample reuse via ``replay_ratio``: the
    defaults give 0.4 updates per vectorized env step with 4096 envs.
    ``wm_window=7`` scores seven predictions from the official eight-frame
    window. Legacy iteration parameters require ``replay_ratio=None``.
    ``buffer_size=None`` stores 1000 rows per env; ``learning_starts=None``
    means ``(wm_window + 2) * num_envs`` transitions. World-model-only warmup
    defaults to zero updates. ``lr_final_ratio < 1`` enables cosine decay.
    ``max_step_reward`` optionally caps bootstrap and critic targets at
    ``max_step_reward / (1 - gamma)``; ``None`` leaves them uncapped.
    """

    total_timesteps: int = 1_000_000
    sim_steps_per_iteration: Optional[int] = None
    updates_per_iteration: Optional[int] = None
    buffer_size: Optional[int] = None
    buffer_device: str = "cuda"
    learning_starts: Optional[int] = None
    gamma: float = 0.99
    amp_dtype: Optional[Literal["bf16"]] = None


@dataclass
class PPOTrainingArgs(PPOParams, EnvRunArgs, CheckpointArgs, PolicyModuleArgs):
    total_timesteps: int = 10_000_000
    num_steps: int = 50
    gamma: float = 0.8
    gae_lambda: float = 0.9
    finite_horizon_gae: bool = False
    # Episode ends whose GAE target bootstraps from final_observation:
    # "truncated" (truncations only; terminations bootstrap 0), "always", "never".
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"

    detach_encoder_on_actor: Optional[bool] = None
    # Decision-mask rollout mode and critic step selection.
    decision_level: bool = False
    per_step_critic: bool = True
    # Policy graph source is observation/policy wiring.
    policy_graph: Optional[str] = None


@dataclass
class VisionPPOTrainingArgs(PPOTrainingArgs, ObservationArgs):
    pass


@dataclass
class GAILTrainingArgs(GAILParams, PPOTrainingArgs):
    """GAIL adds an adversarial discriminator + expert demonstrations on top
    of PPOTrainingArgs's own fields (including ``critic_backbone_type``)."""

    demo_env_id: str = ""
    demo_dataset_backend: str = "d4rl_legacy"
    demo_buffer_size: int = 1_000_000
    demo_batch_size: int = 1024


@dataclass
class RecurrentPPOTrainingArgs(RecurrentPPOParams, PPOTrainingArgs):
    pass


@dataclass
class VisionRecurrentPPOTrainingArgs(RecurrentPPOTrainingArgs, ObservationArgs):
    pass


@dataclass
class TransformerPPOTrainingArgs(TransformerPPOParams, PPOTrainingArgs):
    pass


@dataclass
class DPPOTrainingArgs(DPPOParams, EnvRunArgs, CheckpointArgs):
    """DPPO (Diffusion PPO) fine-tuning. Deliberately does NOT inherit
    ``PPOTrainingArgs``: DPPO's hyperparameters (denoising-chain schedule,
    dual actor/actor_ft learning rates, DPPO's own clip/discount schedule)
    are a different set from standard PPO's, not a superset -- same
    sibling-dataclass reasoning as ``RecurrentPPOTrainingArgs``/
    ``TransformerPPOTrainingArgs``, just with no shared base beyond
    ``EnvRunArgs``/``CheckpointArgs``. State-only (Box observations); no
    ``ObservationArgs``. ``bc_checkpoint`` (required) is the ``DiffusionBC``
    checkpoint DPPO loads its ``actor``/``actor_ft`` weights from."""

    total_timesteps: int = 3_000_000
    bc_checkpoint: str = ""
    num_steps: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    finite_horizon_gae: bool = False
    # Episode ends whose GAE target bootstraps from final_observation:
    # "truncated" (truncations only; terminations bootstrap 0), "always", "never".
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"

    # Evaluate every this many updates (0 disables).
    eval_freq: int = 25
    # Vector-env steps per evaluation.
    num_eval_steps: int = 50


@dataclass
class FlowPPOTrainingArgs(FlowPPOParams, EnvRunArgs, CheckpointArgs):
    """FlowPPO: online PPO fine-tuning for flow-matching policies. Its own
    sibling dataclass, not a ``DPPOTrainingArgs``/``PPOTrainingArgs``
    subclass -- FlowPPO's hyperparameters (SDE schedule, flow-step count,
    a single scalar ``clip_coef``, no denoising-chain schedule) are a
    different set from both. State-only (Box observations); no
    ``ObservationArgs``. Trains ``actor``/``critic`` from scratch, no BC
    checkpoint warm-start (see ``rl_garden/algorithms/flow_ppo.py``'s module
    docstring)."""

    total_timesteps: int = 3_000_000
    num_steps: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    finite_horizon_gae: bool = False
    # Episode ends whose GAE target bootstraps from final_observation:
    # "truncated" (truncations only; terminations bootstrap 0), "always", "never".
    bootstrap_at_done: Literal["always", "never", "truncated"] = "truncated"

    # Evaluate every this many updates (0 disables).
    eval_freq: int = 25
    # Vector-env steps per evaluation.
    num_eval_steps: int = 50


@dataclass
class DiffusionCMDistillOnlineTrainingArgs(DiffusionCMDistillOnlineParams, DPPOTrainingArgs):
    """DiffusionCMDistillOnline: DPPO's online PPO fine-tuning fused with a
    per-iteration LCM one-step distillation step
    (``rl_garden/algorithms/diffusion_cm_distill.py``). A genuine superset of
    ``DPPOTrainingArgs`` (every DPPO field plus the CM student/target/
    optimizer's own), unlike ``DPPOTrainingArgs``'s own deliberate
    non-inheritance from ``PPOTrainingArgs``."""



@dataclass
class VisionTransformerPPOTrainingArgs(TransformerPPOTrainingArgs, ObservationArgs):
    pass



@dataclass
class FlashSACTrainingArgs(FlashSACParams, EnvRunArgs, CheckpointArgs):
    env_id: str = "PickCube-v1"
    num_envs: int = 512
    num_eval_envs: int = 8
    control_mode: str = "pd_ee_delta_pose"
    render_mode: str = "rgb_array"
    buffer_size: int = 10_000_000
    buffer_device: str = "cuda"
    learning_starts: int = 4_000
    batch_size: int = 2048
    gamma: float = 0.99
    training_freq: int = 512
    utd: float = 1.0
    total_timesteps: int = 10_000_000
    asymmetric_obs_dim: int = 0
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    capture_video: bool = False
    video_fps: int = 20


@dataclass
class DAggerTrainingArgs(DAggerParams, EnvRunArgs, CheckpointArgs):
    """DAgger (Ross et al. 2011). A live scripted expert is task-specific (see
    ``expert`` below), so a minimal CLI-selectable set of mock experts is
    what's wired here; a real expert is expected to be supplied by
    constructing ``DAgger`` directly."""

    total_timesteps: int = 100_000
    expert: Literal["zero", "random_uniform"] = "random_uniform"
    demo_buffer_size: int = 100_000
    buffer_device: str = "cuda"
    device: str = "auto"
    beta_rounds: int = 15
    rollout_steps_per_round: int = 1_000
    gradient_steps_per_round: int = 100
    batch_size: int = 256


@dataclass
class PolicyDistillationTrainingArgs(PolicyDistillationParams, EnvRunArgs, CheckpointArgs):
    """Teacher-student on-policy distillation
    (``rl_garden/algorithms/policy_distillation.py``). Dict observations
    only -- the env must expose both a privileged obs group (for the
    teacher) and a realistic obs group (for the student) as named Dict
    keys, selected via ``teacher_obs_keys``/``student_obs_keys``. State-only
    (no vision fields): distillation's motivating use case (sim-to-real
    legged-robot locomotion) is low-dimensional state, not images. v1 loads
    a PPO-trained teacher only -- ``teacher_net_arch`` must match the
    teacher's own architecture (its own checkpoint's obs/action space is
    validated implicitly by shape at load time, non-strict)."""

    total_timesteps: int = 1_000_000
    device: str = "auto"
    teacher_checkpoint: str = ""
    teacher_obs_keys: list[str] = field(default_factory=list)
    student_obs_keys: list[str] = field(default_factory=list)
    teacher_net_arch: tuple[int, ...] = (256, 256, 256)
    num_steps: int = 50


def sac_initial_training_phase_from_args(
    args: SACTrainingArgs,
) -> Optional[InitialTrainingPhase]:
    if args.critic_only_steps <= 0:
        return None
    return InitialTrainingPhase(
        duration_steps=args.critic_only_steps,
        update_actor=False,
        update_critic=True,
        update_encoder=not args.critic_only_freeze_encoder,
        random_action_prob=args.critic_only_random_action_prob,
    )
