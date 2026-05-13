"""Base algorithm: seeding / logger / device / checkpoint I/O.

Minimal counterpart to SB3's ``BaseAlgorithm``. We deliberately keep this
thin so GPU-parallel specifics live in ``OffPolicyAlgorithm``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
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
from rl_garden.common.logger import Logger
from rl_garden.common.utils import get_device, seed_everything
from rl_garden.policies.base import BasePolicy


class BaseAlgorithm(ABC):
    policy: BasePolicy

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

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizers": self._optimizer_state_dicts(),
            "global_step": self._global_step,
            "global_update": self._global_update,
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
