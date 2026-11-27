"""Base algorithm: seeding / logger / device / checkpoint I/O.

Minimal counterpart to SB3's ``BaseAlgorithm``. We deliberately keep this
thin so GPU-parallel specifics live in ``OffPolicyAlgorithm``.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.algorithms.observation_encoders import ObservationEncoderMixin
from rl_garden.common.checkpoint import (
    UNRECORDED,
    SidecarBuffer,
    checkpoint_dict,
    load_checkpoint_file,
    load_replay_buffer_file,
    record_file,
    recorded_file,
    referenced_file,
    replay_buffer_path_for_checkpoint,
    save_checkpoint_file,
    save_replay_buffer_file,
    sidecar_path_for_checkpoint,
    validate_checkpoint_metadata,
    without_local_paths,
)
from rl_garden.common.effective_config import config_digest
from rl_garden.common.eval_metrics import append_masked_episode_metrics
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import STANDARD_UPDATE_MASK, TrainingUpdateMask
from rl_garden.common.utils import get_device, seed_everything
from rl_garden.envs.wrappers.dict_state import VectorizedDictStateWrapper
from rl_garden.policies.base import BasePolicy


def _normalize_env_observation_space(env: Any) -> Any:
    """Boundary normalization (see ``.agents``/observation-redesign plan,
    Layer C): every algorithm sees a Dict observation space and Dict obs
    from here on, never branching on Box vs Dict itself. A bare ``Box``
    ``single_observation_space`` -- a plain gymnasium env or a test double
    that hasn't gone through a backend's own env-construction wrapping --
    gets wrapped here, once. Real backends already emit Dict (each backend
    applies ``DictStateObservationWrapper`` pre-vectorization, or is
    natively Dict), so this is a no-op for them.
    """
    if env is None:
        return env
    space = getattr(env, "single_observation_space", None)
    if not isinstance(space, spaces.Box):
        return env
    return VectorizedDictStateWrapper(env)


class BaseAlgorithm(ObservationEncoderMixin, ABC):
    policy: BasePolicy
    _last_checkpoint_step: int = -1

    # Other algorithm class names whose checkpoints are safe to load, besides
    # the class's own name (``load`` always accepts that one).
    # ``validate_checkpoint_metadata`` accepts a stored algorithm_class that
    # matches the class's own name or *any* entry in this tuple.
    _compatible_checkpoint_algorithms: tuple[str, ...] = ()
    #: Canonical text of the policy graph the policy was built from (set by
    #: the PPO family when a ``policy_graph`` is given); checkpoints must
    #: agree on whether there is one.
    policy_graph_yaml: Optional[str] = None
    #: Why checkpoint evaluation (``rl-garden-eval``) cannot run this
    #: algorithm's policy in an environment, or ``None`` when it can.
    #: Evaluation refuses such a checkpoint with this reason before building
    #: any environment; the checkpoint still rebuilds and loads through the
    #: registry entry's builder.
    checkpoint_eval_unsupported_reason: Optional[str] = None

    #: The run's resolved configuration (what its ``config.json`` holds, as
    #: plain JSON data), set by the training runner and written, without its
    #: absolute paths and command line, to every checkpoint; ``None`` for an
    #: agent built without one.
    run_config: Optional[dict[str, Any]] = None

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
    ) -> None:
        env = _normalize_env_observation_space(env)
        eval_env = _normalize_env_observation_space(eval_env)
        self.env = env
        self.eval_env = eval_env
        self.seed = seed
        self.device = get_device(device)
        self.logger = logger

        seed_everything(seed)
        self._global_step = 0
        self._global_update = 0

    @abstractmethod
    def _setup_model(self) -> None: ...

    @abstractmethod
    def learn(self, total_timesteps: int) -> "BaseAlgorithm": ...

    def _on_training_start(self, total_timesteps: int) -> None:
        """Lifecycle hook called immediately before a training loop starts."""
        del total_timesteps

    def _training_update_mask(self) -> TrainingUpdateMask:
        return STANDARD_UPDATE_MASK

    def _checkpoint_includes_replay_buffer(self) -> bool:
        """Whether ``_save_checkpoint`` writes replay-buffer state. No-op
        default: false. Algorithms with a real replay buffer (e.g.
        ``OffPolicyAlgorithm``) override this to read their own
        CLI-configured ``self.save_replay_buffer``; on-policy algorithms have
        no replay buffer and correctly never need to.
        """
        return False

    def _ddp_extra_broadcast_modules(self) -> list:
        """Extra modules (beyond ``self.policy``, broadcast unconditionally
        by the online runner) that need their rank-0 state broadcast at DDP
        startup. No-op default: none.

        Exists because some algorithms keep trainable state outside
        ``self.policy`` (e.g. SAC's ``alpha_tuner``, see
        ``algorithms/sac_ddp.py``).
        """
        return []

    def _obs_to_policy_device(self, obs):
        """Move CPU-backed env observations to the policy device for inference.

        GPU ManiSkill envs already return tensors on ``self.device`` and this is
        a no-op. This fallback exists for CPU simulator backends such as
        ``physx_cpu``; model training and replay sampling remain CUDA-first.
        """
        if isinstance(obs, dict):
            return {
                k: v if v.device == self.device else v.to(self.device)
                for k, v in obs.items()
            }
        if obs.device == self.device:
            return obs
        return obs.to(self.device)

    def _eval_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.predict(
                self._obs_to_policy_device(obs), deterministic=True
            )

    def _eval_action_and_critic_action(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action actually executed, action the critic evaluated).

        Equal by default; off-policy algorithms that execute a different
        action than the one their critic scores override this to report both.
        """
        action = self._eval_action(obs)
        return action, action

    def _eval_start_hook(self) -> None:
        """Called once before the eval rollout loop starts."""

    def _eval_step_hook(
        self,
        obs_before,
        critic_action: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        infos: dict,
    ) -> None:
        """Called after every eval env step."""

    def _eval_finalize_hook(self) -> dict[str, float]:
        """Extra metrics merged into `_evaluate()`'s return value."""
        return {}

    # --- metric formatting / logging ---

    @staticmethod
    def _first_metric(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return float("nan")

    @staticmethod
    def _fmt_metric(value: float) -> str:
        return "nan" if value != value else f"{value:.4f}"

    def _log_eval_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is None:
            return
        for key, value in metrics.items():
            self.logger.add_scalar(f"eval/{key}", value, step)

    def _log_rollout_metric(self, key: str, value: float, step: int) -> None:
        if self.logger is None:
            return
        self.logger.add_scalar(f"train/{key}", value, step)

    def _log_env_log_scalars(self, infos: dict, step: int) -> None:
        """Log an env step's optional ``infos["log"]`` scalar aggregates as
        ``env/<key>``, with one host transfer for all tensor values. Values
        that are neither a real number nor a 1-element tensor are skipped."""
        if self.logger is None or "log" not in infos:
            return
        env_log = infos["log"]
        values = {key: float(value) for key, value in env_log.items() if isinstance(value, Real)}
        tensor_keys = [
            key
            for key, value in env_log.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        ]
        if tensor_keys:
            stacked = torch.stack([env_log[key].detach().reshape(()).float() for key in tensor_keys])
            values.update(zip(tensor_keys, stacked.tolist()))
        for key, value in values.items():
            self.logger.add_scalar(f"env/{key}", value, step)

    def _log_update_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Extension point: log the metrics dict ``train()`` returned, at
        ``step`` (the global step), on the training loop's logging steps
        (``log_freq``). Logs nothing without a logger. Subclasses log under
        their own prefix (``OnPolicyAlgorithm``: ``losses/<key>``); an
        override that adds metrics calls ``super()`` so the base keys stay
        logged."""
        if self.logger is None:
            return
        self.logger.log_metrics(metrics, step)

    # --- evaluation ---

    def _evaluate(self) -> dict[str, float]:
        if self.eval_env is None:
            return {}
        self.policy.eval()
        obs, _ = self.eval_env.reset()
        self._eval_start_hook()
        metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
        for _ in range(self.num_eval_steps):
            with torch.no_grad():
                env_action, critic_action = self._eval_action_and_critic_action(obs)
                obs_before = obs
                obs, rewards, terminations, truncations, infos = self.eval_env.step(
                    env_action
                )
                self._eval_step_hook(
                    obs_before, critic_action, rewards, terminations, truncations, infos
                )
                append_masked_episode_metrics(metrics, infos)
        self.policy.train()
        out: dict[str, float] = {}
        for k, vs in metrics.items():
            out[k] = float(torch.cat(vs).float().mean().item())
        out.update(self._eval_finalize_hook())
        return out

    # --- checkpointing ---

    def _optimizer_names(self) -> tuple[str, ...]:
        return (
            "q_optimizer",
            "actor_optimizer",
            "alpha_optimizer",
            "cql_alpha_optimizer",
        )

    def _optimizer_state_dicts(self) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for name in self._optimizer_names():
            optimizer = getattr(self, name, None)
            if optimizer is not None:
                states[name] = optimizer.state_dict()
        return states

    def _load_optimizer_state_dicts(self, states: dict[str, Any]) -> None:
        for name, state in states.items():
            optimizer = getattr(self, name, None)
            if optimizer is not None:
                optimizer.load_state_dict(state)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": str(self.device),
        }

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_checkpoint_state(self, state: dict[str, Any]) -> None:
        del state

    def _training_state_dict(self) -> dict[str, Any]:
        return {}

    def _load_training_state_dict(self, state: dict[str, Any]) -> None:
        del state

    @property
    def global_update(self) -> int:
        """Read-only view of ``_global_update``, restored by ``load_state_dict``.

        Exposed for callers outside the algorithm hierarchy (e.g. real-world
        ``LearnerLoop``) that need a training-progress counter which stays
        monotonic across a checkpoint resume, without reaching into the
        underscore-prefixed internal field.
        """
        return self._global_update

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizers": self._optimizer_state_dicts(),
            "global_step": self._global_step,
            "global_update": self._global_update,
            "training_state": self._training_state_dict(),
            "extra": self._extra_checkpoint_state(),
        }

    def load_state_dict(
        self,
        sd: dict[str, Any],
        strict: bool = True,
        load_optimizers: bool = True,
    ) -> None:
        self.policy.load_state_dict(sd["policy"], strict=strict)
        self._global_step = int(sd.get("global_step", 0))
        self._global_update = int(sd.get("global_update", 0))
        self._load_training_state_dict(sd.get("training_state", {}))
        self._load_extra_checkpoint_state(sd.get("extra", {}))
        if load_optimizers:
            self._load_optimizer_state_dicts(sd.get("optimizers", {}))

    def _checkpoint_path(self, name: str) -> Path:
        assert self.checkpoint_dir is not None
        return Path(self.checkpoint_dir) / name

    def _save_checkpoint(self, name: str) -> None:
        if self.checkpoint_dir is None:
            return
        self.save(
            self._checkpoint_path(name),
            include_replay_buffer=self._checkpoint_includes_replay_buffer(),
        )

    def _maybe_save_final_checkpoint(self) -> None:
        if self.checkpoint_dir is not None and self.save_final_checkpoint:
            self._save_checkpoint("final.pt")

    def _maybe_save_periodic_checkpoint(self, previous_step: int) -> None:
        if self.checkpoint_dir is None or self.checkpoint_freq <= 0:
            return
        if self._global_step // self.checkpoint_freq <= previous_step // self.checkpoint_freq:
            return
        if self._global_step == self._last_checkpoint_step:
            return
        self._save_checkpoint(f"checkpoint_{self._global_step}.pt")
        self._last_checkpoint_step = self._global_step

    def _sidecar_buffers(self) -> tuple[SidecarBuffer, ...]:
        """Buffers saved beside the checkpoint together with the replay
        buffer. ``save(include_replay_buffer=True)`` names each written file at
        its ``reference_key`` under ``state["extra"]`` (the subclass's
        ``_extra_checkpoint_state`` writes that slot as ``None``);
        ``load(load_replay_buffer=True)`` asks again after the state is
        restored. Extend with ``(*super()._sidecar_buffers(), SidecarBuffer(...))``."""
        return ()

    def _on_checkpoint_read(
        self, checkpoint: dict[str, Any], *, replay_path: Optional[Path]
    ) -> None:
        """Called by ``load()`` once the file is read and its metadata
        validated, before any state or buffer is restored. ``replay_path`` is
        the replay file ``load()`` is about to read, ``None`` when it reads
        none. An override calls ``super()`` first, raises to reject the
        checkpoint (whatever ``strict`` is), and may fill in, in ``checkpoint``
        itself, what an older checkpoint does not record."""
        del checkpoint, replay_path

    def save(self, path: str | Path, include_replay_buffer: bool = False) -> Path:
        """Save model/optimizer/train state to ``path``.

        When ``include_replay_buffer`` is true, replay state is written to a
        sibling file (for example ``checkpoint_1000.pt`` +
        ``replay_buffer_1000.pt``), and each buffer of ``_sidecar_buffers`` to
        its own (``<prefix>_1000.pt``); the checkpoint names them. The
        checkpoint file holds no absolute path and no command line
        (``without_local_paths``).
        """
        checkpoint_path = Path(path)
        replay_path = (
            replay_buffer_path_for_checkpoint(checkpoint_path)
            if include_replay_buffer and hasattr(self, "replay_buffer")
            else None
        )
        state = self.state_dict()
        sidecar_files: list[tuple[Path, Any]] = []
        if include_replay_buffer:
            for sidecar in self._sidecar_buffers():
                if sidecar.buffer is None:
                    continue
                sidecar_path = sidecar_path_for_checkpoint(checkpoint_path, sidecar.prefix)
                record_file(state["extra"], sidecar.reference_key, sidecar_path.name)
                sidecar_files.append((sidecar_path, sidecar.buffer))
        checkpoint = checkpoint_dict(
            algorithm_class=type(self).__name__,
            global_step=self._global_step,
            global_update=self._global_update,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            hyperparameters=self._checkpoint_metadata(),
            state=state,
            replay_buffer_path=replay_path.name if replay_path is not None else None,
        )
        if self.run_config is not None:
            checkpoint["metadata"]["run_config"] = self.run_config
            checkpoint["metadata"]["run_config_digest"] = config_digest(self.run_config)
        save_checkpoint_file(checkpoint_path, without_local_paths(checkpoint))
        if replay_path is not None:
            save_replay_buffer_file(replay_path, self.replay_buffer)
        for sidecar_path, buffer in sidecar_files:
            save_replay_buffer_file(sidecar_path, buffer)
        return checkpoint_path

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_replay_buffer: bool = True,
        load_optimizers: bool = True,
    ) -> "BaseAlgorithm":
        """Load a checkpoint in-place into an already constructed agent.

        With ``load_replay_buffer``, the replay file and the files of
        ``_sidecar_buffers`` the checkpoint names are read too. A named file
        must exist; ``None`` recorded means the save wrote none; a checkpoint
        without the replay reference reads ``replay_buffer_<suffix>.pt`` if it
        exists.
        """
        checkpoint_path = Path(path)
        checkpoint = load_checkpoint_file(checkpoint_path, map_location=self.device)
        validate_checkpoint_metadata(
            checkpoint,
            algorithm_class=type(self).__name__,
            compatible_algorithms=(
                type(self).__name__,
                *self._compatible_checkpoint_algorithms,
            ),
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            strict=strict,
            policy_graph=self.policy_graph_yaml,
        )
        replay_reference = recorded_file(checkpoint.get("metadata", {}), ("replay_buffer_path",))
        conventional_replay_path = replay_buffer_path_for_checkpoint(checkpoint_path)
        replay_path = None
        if load_replay_buffer and hasattr(self, "replay_buffer"):
            replay_path = referenced_file(
                checkpoint_path,
                replay_reference,
                unrecorded=conventional_replay_path,
                description="replay buffer",
            )
        self._on_checkpoint_read(checkpoint, replay_path=replay_path)
        self.load_state_dict(checkpoint["state"], strict=strict, load_optimizers=load_optimizers)
        if replay_path is not None:
            load_replay_buffer_file(replay_path, self.replay_buffer, strict=strict)
        if load_replay_buffer:
            if replay_reference is UNRECORDED:
                named_replay_path: Optional[Path] = conventional_replay_path
            elif replay_reference is None:
                named_replay_path = None
            else:
                named_replay_path = checkpoint_path.parent / replay_reference
            self._load_sidecar_buffers(
                checkpoint_path, checkpoint, named_replay_path, strict=strict
            )
        return self

    def _load_sidecar_buffers(
        self,
        checkpoint_path: Path,
        checkpoint: dict[str, Any],
        replay_path: Optional[Path],
        *,
        strict: bool,
    ) -> None:
        """``replay_path``: the replay file the checkpoint names, whether or
        not it exists; ``None`` when it names none."""
        extra = checkpoint["state"].get("extra", {})
        for sidecar in self._sidecar_buffers():
            description = sidecar.prefix.replace("_", " ")
            reference = recorded_file(extra, sidecar.reference_key)
            beside_replay = None
            if sidecar.unrecorded_beside_replay and replay_path is not None:
                beside_replay = replay_path.with_name(
                    replay_path.name.replace("replay_buffer_", f"{sidecar.prefix}_", 1)
                )
                if beside_replay == replay_path:
                    beside_replay = None
            sidecar_path = referenced_file(
                checkpoint_path, reference, unrecorded=beside_replay, description=description
            )
            if sidecar_path is not None:
                sidecar.restore(sidecar_path, strict)
            elif reference is UNRECORDED and beside_replay is not None and replay_path.exists():
                warnings.warn(
                    f"{checkpoint_path} does not record whether it was saved with a "
                    f"{description} and {beside_replay} does not exist: continuing with an "
                    f"empty {description}.",
                    stacklevel=3,
                )

    def save_replay_buffer(self, path: str | Path) -> Path:
        return save_replay_buffer_file(path, self.replay_buffer)

    def load_replay_buffer(self, path: str | Path, strict: bool = True) -> None:
        load_replay_buffer_file(path, self.replay_buffer, strict=strict)
