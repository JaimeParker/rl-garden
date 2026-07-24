"""Base algorithm: seeding / logger / device / checkpoint I/O.

Minimal counterpart to SB3's ``BaseAlgorithm``. We deliberately keep this
thin so GPU-parallel specifics live in ``OffPolicyAlgorithm``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden.common.checkpoint import (
    checkpoint_dict,
    load_checkpoint_file,
    load_replay_buffer_file,
    replay_buffer_path_for_checkpoint,
    save_checkpoint_file,
    save_replay_buffer_file,
    validate_checkpoint_metadata,
)
from rl_garden.common.eval_metrics import append_masked_episode_metrics
from rl_garden.common.logger import Logger
from rl_garden.common.training_phase import STANDARD_UPDATE_MASK, TrainingUpdateMask
from rl_garden.common.utils import get_device, seed_everything
from rl_garden.policies.base import BasePolicy


class BaseAlgorithm(ABC):
    policy: BasePolicy

    # Subclasses may list parent algorithm class names whose checkpoints are
    # safe to load.  ``validate_checkpoint_metadata`` will accept a stored
    # algorithm_class that matches *any* entry in this tuple.
    _compatible_checkpoint_algorithms: tuple[str, ...] = ()

    def __init__(
        self,
        env: Any,
        eval_env: Optional[Any] = None,
        seed: int = 1,
        device: str | torch.device = "auto",
        logger: Optional[Logger] = None,
    ) -> None:
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

        Equal by default; off-policy/residual algorithms that execute a
        different action than the one their critic scores (e.g. residual RL)
        override this to report both.
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
                if "final_info" in infos:
                    append_masked_episode_metrics(
                        metrics,
                        infos["final_info"]["episode"],
                        infos.get("_final_info"),
                    )
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

    def save(self, path: str | Path, include_replay_buffer: bool = False) -> Path:
        """Save model/optimizer/train state to ``path``.

        When ``include_replay_buffer`` is true, replay state is written to a
        sibling file (for example ``checkpoint_1000.pt`` +
        ``replay_buffer_1000.pt``) and referenced from the model checkpoint.
        """
        checkpoint_path = Path(path)
        replay_path = (
            replay_buffer_path_for_checkpoint(checkpoint_path)
            if include_replay_buffer and hasattr(self, "replay_buffer")
            else None
        )
        checkpoint = checkpoint_dict(
            algorithm_class=type(self).__name__,
            global_step=self._global_step,
            global_update=self._global_update,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            hyperparameters=self._checkpoint_metadata(),
            state=self.state_dict(),
            replay_buffer_path=replay_path.name if replay_path is not None else None,
        )
        save_checkpoint_file(checkpoint_path, checkpoint)
        if replay_path is not None:
            save_replay_buffer_file(replay_path, self.replay_buffer)
        return checkpoint_path

    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_replay_buffer: bool = True,
        load_optimizers: bool = True,
    ) -> "BaseAlgorithm":
        """Load a checkpoint in-place into an already constructed agent."""
        checkpoint_path = Path(path)
        checkpoint = load_checkpoint_file(checkpoint_path, map_location=self.device)
        validate_checkpoint_metadata(
            checkpoint,
            algorithm_class=type(self).__name__,
            compatible_algorithms=self._compatible_checkpoint_algorithms,
            observation_space=self.env.single_observation_space,
            action_space=self.env.single_action_space,
            strict=strict,
        )
        self.load_state_dict(checkpoint["state"], strict=strict, load_optimizers=load_optimizers)

        replay_ref = checkpoint.get("metadata", {}).get("replay_buffer_path")
        if load_replay_buffer and hasattr(self, "replay_buffer"):
            replay_path = (
                checkpoint_path.parent / replay_ref
                if replay_ref is not None
                else replay_buffer_path_for_checkpoint(checkpoint_path)
            )
            if replay_path.exists():
                load_replay_buffer_file(replay_path, self.replay_buffer, strict=strict)
            elif replay_ref is not None:
                raise FileNotFoundError(
                    f"Checkpoint references replay buffer {replay_path}, but it does not exist."
                )
        return self

    def save_replay_buffer(self, path: str | Path) -> Path:
        return save_replay_buffer_file(path, self.replay_buffer)

    def load_replay_buffer(self, path: str | Path, strict: bool = True) -> None:
        load_replay_buffer_file(path, self.replay_buffer, strict=strict)
