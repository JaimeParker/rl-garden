"""Replay buffer interface.

Replay buffers expose torch tensors throughout the training path. The default
storage is GPU-native; selected vision buffers can optionally use CPU memmaps
and move sampled batches directly to the policy device.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch

from rl_garden.common.types import Obs, ReplayBufferSample


def storage_state(storage: Any) -> Any:
    """``storage`` (a tensor, or a ``DictArray``/dict of them) as plain dicts of tensors."""
    if isinstance(storage, torch.Tensor):
        return storage
    tree = storage if isinstance(storage, dict) else storage.data
    return {key: storage_state(value) for key, value in tree.items()}


def restored_storage(current: Any, saved: Any, name: str) -> Any:
    """``saved`` (as ``storage_state`` returned it) in the form of ``current``,
    after checking that both have the same keys and shapes."""
    if isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor) or current.shape != saved.shape:
            saved_shape = tuple(saved.shape) if isinstance(saved, torch.Tensor) else saved
            raise ValueError(
                f"Replay buffer tensor shape mismatch for {name!r}: "
                f"checkpoint={saved_shape}, current={tuple(current.shape)}"
            )
        return saved
    tree = current if isinstance(current, dict) else current.data
    if not isinstance(saved, dict) or set(tree) != set(saved):
        saved_keys = sorted(saved) if isinstance(saved, dict) else saved
        raise ValueError(
            f"Replay buffer {name!r} keys do not match: "
            f"checkpoint={saved_keys}, current={sorted(tree)}"
        )
    for key in tree:
        tree[key] = restored_storage(tree[key], saved[key], key)
    return current


class BaseReplayBuffer(ABC):
    """Minimal interface every replay buffer implements.

    Layout: ``(per_env_buffer_size, num_envs, ...)`` — one slot per env per
    timestep, indexed circularly.

    ``state_dict()``/``load_state_dict()`` hold everything ``sample()`` and the
    next ``add()`` read; a subclass with more state than fixed-shape storage
    (``_state_attrs``) extends both.
    """

    num_envs: int
    per_env_buffer_size: int
    pos: int
    full: bool
    storage_device: torch.device
    sample_device: torch.device

    #: ``_state_attrs`` that files written by earlier versions lack and that
    #: cannot be rebuilt from the rest. Such a file loads only over a buffer that
    #: already holds its rows (an opened mmap store, a reloaded dataset), whose
    #: values are then kept.
    _window_state_absent_from_old_files: tuple[str, ...] = ()
    #: The same, for attributes whose initial values are a valid substitute.
    _columns_absent_from_old_files: tuple[str, ...] = ()
    #: A sample reads several consecutive rows of one env column (returns, next
    #: actions, windows), so a dataset is loaded as one stretch of trajectories
    #: per column (``add_flat_transitions``).
    rows_form_trajectories: bool = False

    @abstractmethod
    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSample: ...

    @property
    def size(self) -> int:
        """Number of transitions stored per env."""
        return self.per_env_buffer_size if self.full else self.pos

    def __len__(self) -> int:
        """Total transitions available across all envs."""
        return self.size * self.num_envs

    @property
    def sampleable_size(self) -> int:
        """Total transitions currently eligible for sampling.

        Defaults to ``len(self)``; buffers that can hold not-yet-sampleable
        transitions (e.g. an MC buffer's incomplete trailing trajectories)
        override this.
        """
        return len(self)

    def cut_running_episodes(self) -> None:
        """Every env was reset outside ``env.step``: the episodes still running at
        the last stored row end there, truncated, and the next ``add()`` starts
        new ones. ``OffPolicyAlgorithm.learn()`` calls this after its
        ``env.reset()``; a loop that resets the env itself while it fills a buffer
        calls it too.

        A class that computes returns, bootstrap observations or observation
        windows from several rows overrides this so that none of them takes the
        last row before the cut and the first row after it for consecutive steps
        of one episode. A stored ``dones`` flag never changes. Nothing to do for a
        sample that holds the fields of one row only.
        """

    def _state_attrs(self) -> tuple[str, ...]:
        """Attributes holding fixed-shape storage (tensor, ``DictArray`` or dict
        of tensors). A leading underscore is dropped in the ``state_dict`` key."""
        return ("obs", "actions", "rewards", "dones")

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"pos": int(self.pos), "full": bool(self.full)}
        for attr in self._state_attrs():
            state[attr.lstrip("_")] = storage_state(getattr(self, attr))
        if hasattr(self, "gamma"):
            state["gamma"] = float(self.gamma)
        return state

    def _check_missing_state(
        self, missing: list[str], state: dict[str, Any], strict: bool
    ) -> None:
        window_state = [a for a in missing if a in self._window_state_absent_from_old_files]
        columns = [a for a in missing if a in self._columns_absent_from_old_files]
        required = [a for a in missing if a not in (*window_state, *columns)]
        holds_the_rows = len(self) > 0 and (self.pos, self.full) == (state["pos"], state["full"])
        if strict and required:
            names = ", ".join(repr(attr.lstrip("_")) for attr in required)
            raise ValueError(f"Replay buffer checkpoint is missing {names}.")
        if strict and window_state and not holds_the_rows:
            names = [attr.lstrip("_") for attr in window_state]
            raise ValueError(
                f"The replay buffer checkpoint was written without {names}, which "
                f"{type(self).__name__} needs to sample its transitions and cannot rebuild. "
                "Resume without the replay buffer instead."
            )
        if window_state or columns:
            names = [attr.lstrip("_") for attr in (*window_state, *columns)]
            warnings.warn(
                f"The replay buffer checkpoint was written without {names}; "
                f"{type(self).__name__} keeps its current values for them.",
                stacklevel=4,
            )

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        """Adopt the tensors of ``state``, which must be on ``storage_device``."""
        missing = [attr for attr in self._state_attrs() if attr.lstrip("_") not in state]
        self._check_missing_state(missing, state, strict)
        for attr in self._state_attrs():
            if attr not in missing:
                key = attr.lstrip("_")
                setattr(self, attr, restored_storage(getattr(self, attr), state[key], key))
        self.pos = int(state["pos"])
        self.full = bool(state["full"])
        if hasattr(self, "gamma") and "gamma" in state:
            self.gamma = float(state["gamma"])

    def _advance(self) -> None:
        """Advance the circular write cursor, wrapping when the per-env buffer fills."""
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
