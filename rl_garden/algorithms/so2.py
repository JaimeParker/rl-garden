"""SO2: SAC with a large critic ensemble and Bellman-target action smoothing.

Ported from ``3rd_party/SO2``'s ``origin/code`` branch (Zhang et al. 2024,
arXiv:2312.07685, "A Perspective of Q-value Estimation on Offline-to-Online
RL"). The checked-out ``main`` branch is a red herring -- it's a thin 2-file
stub against stock pip ``DI-engine@v0.5.1``, and none of its distinguishing
config keys (``noise``, ``only_value``, ``concat_online_ratio``,
``actor_update_freq``, ...) exist anywhere in that DI-engine version
(verified by pulling ``sac.py``/``edac.py`` at that tag and grepping for zero
matches). The real algorithm lives in ``ding/policy/edac.py`` on
``origin/code``, read via ``git show origin/code:...`` -- the
``3rd_party/SO2`` working tree itself was never touched or checked out.

SO2 is plain multi-critic SAC: a large random critic ensemble with a
min-over-ensemble entropy-corrected Bellman target -- **no diversity-loss
term at all**, unlike CORL-style ``EDAC`` already in this repo. Its one
genuinely new mechanism is a TD3-style target-smoothing regularizer: clamped
Gaussian noise added to the *next_action* used only for the Bellman-target
computation (not the entropy log-prob). ``n_critics``/``policy_frequency``
are already generic ``SACCore``/``SAC`` kwargs and map 1:1 to upstream's
ensemble size/``actor_update_freq`` -- no new code needed for those.

``SO2Core`` and its shell build directly on ``SAC``/``OfflineSAC`` (the
literature-subtyping exception in ``adding-algorithm.md`` Part C, the same
precedent ``WSRL``/``Off2OnCalQL`` use to build on ``CQL``'s shell rather
than re-deriving ``OffPolicyAlgorithm`` from scratch): SO2 needs no new
networks and no new loss terms beyond the target-smoothing override below.

Box (state) observations only for this port -- matches every env this
targets (D4RL MuJoCo locomotion, AntMaze; both use rl-garden's existing
``d4rl_legacy`` env/dataset backend, no new backend work needed). Dict/image
support can be added later following IQL/CalQL's own precedent if needed.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.algorithms.off2on import Off2OnReplayMixin
from rl_garden.algorithms.offline import OfflineEnvSpec
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer


class _MirrorAddMixin:
    """Buffer mixin: every ``add()`` also writes into a sibling buffer.

    Reproduces upstream's offline-buffer FIFO churn
    (``serial_entry_offline2online.py:78-86`` on ``origin/code``): the
    offline-side pool is an ordinary fixed-capacity ring buffer that absorbs
    newly collected online transitions after the online switch, evicting its
    oldest entries once full (sizing it to the loaded dataset's transition
    count, via ``--buffer_size``, reproduces this exactly). A real buffer
    subclass -- not a thin proxy -- so checkpointing (which reads the
    buffer's own tensors/``.pos``/``.full`` directly) is unaffected.
    """

    _mirror_into: Optional[Any] = None

    def add(self, *args: Any, **kwargs: Any) -> None:
        super().add(*args, **kwargs)
        if self._mirror_into is not None:
            self._mirror_into.add(*args, **kwargs)


class _MirroringTensorReplayBuffer(_MirrorAddMixin, TensorReplayBuffer):
    pass


class SO2Core:
    """Shared SO2 behavior: target-action smoothing + mirroring buffer."""

    def _init_so2_params(
        self,
        *,
        target_smoothing_noise_std: float = 0.3,
        target_smoothing_noise_clip_min: float = -0.6,
        target_smoothing_noise_clip_max: float = 0.6,
    ) -> None:
        if target_smoothing_noise_std < 0:
            raise ValueError(
                "target_smoothing_noise_std must be >= 0, got "
                f"{target_smoothing_noise_std}."
            )
        if target_smoothing_noise_clip_min > target_smoothing_noise_clip_max:
            raise ValueError(
                "target_smoothing_noise_clip_min must be <= "
                "target_smoothing_noise_clip_max, got "
                f"{target_smoothing_noise_clip_min} > {target_smoothing_noise_clip_max}."
            )
        self.target_smoothing_noise_std = target_smoothing_noise_std
        self.target_smoothing_noise_clip_min = target_smoothing_noise_clip_min
        self.target_smoothing_noise_clip_max = target_smoothing_noise_clip_max

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_metadata(),
            "target_smoothing_noise_std": self.target_smoothing_noise_std,
            "target_smoothing_noise_clip_min": self.target_smoothing_noise_clip_min,
            "target_smoothing_noise_clip_max": self.target_smoothing_noise_clip_max,
        }

    def _smooth_target_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.target_smoothing_noise_std <= 0:
            return action
        noise = torch.randn_like(action) * self.target_smoothing_noise_std
        noise = noise.clamp(
            self.target_smoothing_noise_clip_min,
            self.target_smoothing_noise_clip_max,
        )
        return (action + noise).clamp(-1.0, 1.0)

    def _target_action_log_prob(
        self, data
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Only the target-Q lookup action is perturbed -- next_log_prob (the
        # entropy correction) is computed from the un-noised action, matching
        # upstream's edac.py:333-351 exactly (noise is added to next_action
        # only after next_log_prob has already been computed from it).
        next_action, next_log_prob, next_actor_features = (
            super()._target_action_log_prob(data)
        )
        next_action = self._smooth_target_action(next_action)
        return next_action, next_log_prob, next_actor_features

    def _build_replay_buffer(self):
        obs_space = self.env.single_observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(
                f"SO2 supports Box observation spaces only, got {type(obs_space)}."
            )
        buffer = _MirroringTensorReplayBuffer(
            observation_space=obs_space,
            action_space=self.env.single_action_space,
            num_envs=self.num_envs,
            buffer_size=self.buffer_size,
            storage_device=self.buffer_device,
            sample_device=self.device,
        )
        # None before the online switch (and for the standalone offline SO2
        # class below, which has no `offline_replay_buffer` attribute at all
        # -- Off2OnReplayMixin isn't in its MRO); the real, already-assigned
        # offline buffer once Off2OnReplayMixin.switch_to_online_mode has run.
        buffer._mirror_into = getattr(self, "offline_replay_buffer", None)
        return buffer


class SO2(SO2Core, OfflineSAC):
    """Pure offline SO2: multi-critic SAC + Bellman-target action smoothing."""

    _compatible_checkpoint_algorithms = ("SO2",)

    def __init__(
        self,
        env: OfflineEnvSpec,
        *args: Any,
        target_smoothing_noise_std: float = 0.3,
        target_smoothing_noise_clip_min: float = -0.6,
        target_smoothing_noise_clip_max: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, *args, **kwargs)
        self._init_so2_params(
            target_smoothing_noise_std=target_smoothing_noise_std,
            target_smoothing_noise_clip_min=target_smoothing_noise_clip_min,
            target_smoothing_noise_clip_max=target_smoothing_noise_clip_max,
        )
        # OfflineSAC.__init__ never sets self.backup_entropy, which
        # SACCore._backup_entropy_enabled() requires (a pre-existing gap in
        # OfflineSAC itself -- see EDAC's identical fix/comment in edac.py).
        # SO2's own reference entropy-corrects its critic target
        # (edac.py:356 on origin/code, `target_q_value - alpha*next_log_prob`),
        # so this is also the numerically correct value for SO2 specifically.
        self.backup_entropy = True


class _SO2RolloutTrainingShell(Off2OnReplayMixin, SO2Core, SAC):
    """Internal rollout/eval shell that wires ``SO2Core`` into ``SAC``.

    Generic offline->online transition mechanics (replay-buffer switching,
    mixed-batch sampling, checkpoint/probe/logging plumbing) are inherited
    from ``Off2OnReplayMixin``. SO2 needs no algorithm-specific override at
    the online switch beyond the mirroring buffer already wired through
    ``SO2Core._build_replay_buffer`` -- neither
    ``_apply_online_regularizer_override`` nor ``_offline_probe_metrics`` is
    overridden here.

    .. warning::
       **Do not instantiate this class directly.** It exists only to back
       :class:`~rl_garden.algorithms.Off2OnSO2`. For standalone offline SO2
       pretraining use :class:`SO2`. The shape and arguments of this shell
       may change without notice.
    """

    def __init__(
        self,
        *args: Any,
        target_smoothing_noise_std: float = 0.3,
        target_smoothing_noise_clip_min: float = -0.6,
        target_smoothing_noise_clip_max: float = 0.6,
        offline_sampling: str = "with_replace",
        **kwargs: Any,
    ) -> None:
        self._init_so2_params(
            target_smoothing_noise_std=target_smoothing_noise_std,
            target_smoothing_noise_clip_min=target_smoothing_noise_clip_min,
            target_smoothing_noise_clip_max=target_smoothing_noise_clip_max,
        )
        self._init_off2on_params(offline_sampling=offline_sampling)
        super().__init__(*args, **kwargs)
