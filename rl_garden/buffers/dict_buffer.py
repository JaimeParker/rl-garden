"""Dict replay buffer for ``{rgb, depth, state, ...}`` observations.

Built on a ``DictArray`` container lifted from ManiSkill's
``examples/baselines/sac/sac_rgbd.py``. Each observation key gets its own
GPU tensor with shape ``(per_env_buffer_size, num_envs, *space.shape)`` and
the appropriate dtype (uint8 for images, float32 for state, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers._sampling import WithoutReplaceSamplerMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.mmap_storage import (
    MmapMode,
    MmapTensorStore,
    space_metadata,
)
from rl_garden.common.types import ReplayBufferSample, TensorDict


def _resolve_dtype(np_dtype) -> torch.dtype:
    if np_dtype in (np.float32, np.float64):
        return torch.float32
    if np_dtype == np.uint8:
        return torch.uint8
    if np_dtype == np.int16:
        return torch.int16
    if np_dtype == np.int32:
        return torch.int32
    # Fallback — let torch complain if it's not convertible.
    return torch.as_tensor(np.empty((), dtype=np_dtype)).dtype


def _tensor_to_device(
    value: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> torch.Tensor:
    if (
        pin_memory
        and non_blocking
        and device.type == "cuda"
        and value.device.type == "cpu"
    ):
        try:
            value = value.pin_memory()
        except RuntimeError:
            # Some CPU-only builds expose pin_memory but cannot allocate pinned
            # pages. Fall back to the normal blocking copy in that case.
            non_blocking = False
    return value.to(device, non_blocking=non_blocking)


def _tree_to_device(
    value: Any,
    device: torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _tree_to_device(
                item,
                device,
                non_blocking=non_blocking,
                pin_memory=pin_memory,
            )
            for key, item in value.items()
        }
    return _tensor_to_device(
        value,
        device,
        non_blocking=non_blocking,
        pin_memory=pin_memory,
    )


class DictArray:
    """A pytree-lite: nested dict of same-shape torch tensors.

    Supports indexing with integers / tensors (returns plain ``dict``) and
    string keys (returns underlying tensor / sub-DictArray). ``__setitem__``
    with an int index writes a whole time-slice across all keys.
    """

    def __init__(
        self,
        buffer_shape: tuple[int, ...],
        element_space: spaces.Dict | None,
        data_dict: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        mmap_store: Optional[MmapTensorStore] = None,
        mmap_path: tuple[str, ...] = (),
    ) -> None:
        self.buffer_shape = buffer_shape
        if data_dict is not None:
            self.data = data_dict
            return

        assert isinstance(element_space, spaces.Dict)
        self.data: dict[str, Any] = {}
        for k, v in element_space.items():
            if isinstance(v, spaces.Dict):
                self.data[k] = DictArray(
                    buffer_shape,
                    v,
                    device=device,
                    mmap_store=mmap_store,
                    mmap_path=(*mmap_path, k),
                )
            elif mmap_store is not None:
                dtype = _resolve_dtype(v.dtype)
                self.data[k] = mmap_store.tensor(
                    (*mmap_path, k),
                    shape=buffer_shape + tuple(v.shape),
                    dtype=dtype,
                )
            else:
                dtype = _resolve_dtype(v.dtype)
                self.data[k] = torch.zeros(
                    buffer_shape + tuple(v.shape), dtype=dtype, device=device
                )

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self) -> tuple[int, ...]:
        return self.buffer_shape


class DictReplayBuffer(WithoutReplaceSamplerMixin, BaseReplayBuffer):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        mmap_dir: Optional[str | Path] = None,
        mmap_mode: MmapMode = "create",
    ) -> None:
        assert isinstance(observation_space, spaces.Dict), (
            "DictReplayBuffer requires a Dict observation space."
        )
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        if mmap_dir is not None and self.storage_device.type != "cpu":
            raise ValueError(
                "mmap replay buffers require CPU storage; "
                "pass buffer_device='cpu'"
            )

        shape = (self.per_env_buffer_size, num_envs)
        self._mmap_store = None
        self._cursor = None
        if mmap_dir is not None:
            self._mmap_store = MmapTensorStore(
                mmap_dir,
                mode=mmap_mode,
                manifest={
                    "buffer_class": type(self).__name__,
                    "num_envs": num_envs,
                    "buffer_size": buffer_size,
                    "per_env_buffer_size": self.per_env_buffer_size,
                    "observation_space": space_metadata(
                        observation_space, dtype_resolver=_resolve_dtype
                    ),
                    "action_space": space_metadata(
                        action_space, dtype_resolver=_resolve_dtype
                    ),
                },
            )
            self._cursor = self._mmap_store.tensor(
                ("metadata", "cursor"),
                shape=(2,),
                dtype=torch.int64,
            )
            self.pos = int(self._cursor[0].item())
            self.full = bool(self._cursor[1].item())
        else:
            self.pos = 0
            self.full = False

        self.obs = DictArray(
            shape,
            observation_space,
            device=self.storage_device,
            mmap_store=self._mmap_store,
            mmap_path=("obs",),
        )
        self.next_obs = DictArray(
            shape,
            observation_space,
            device=self.storage_device,
            mmap_store=self._mmap_store,
            mmap_path=("next_obs",),
        )
        if self._mmap_store is None:
            self.actions = torch.zeros(
                shape + tuple(action_space.shape), device=self.storage_device
            )
            self.rewards = torch.zeros(shape, device=self.storage_device)
            self.dones = torch.zeros(shape, device=self.storage_device)
        else:
            self.actions = self._mmap_store.tensor(
                ("actions",),
                shape=shape + tuple(action_space.shape),
                dtype=torch.float32,
            )
            self.rewards = self._mmap_store.tensor(
                ("rewards",), shape=shape, dtype=torch.float32
            )
            self.dones = self._mmap_store.tensor(
                ("dones",), shape=shape, dtype=torch.float32
            )

        if self.pos < 0 or self.pos >= self.per_env_buffer_size:
            raise ValueError(f"Invalid mmap replay cursor position: {self.pos}")

    def _persist_cursor(self) -> None:
        if self._cursor is not None:
            self._cursor[0] = self.pos
            self._cursor[1] = int(self.full)

    def flush(self) -> None:
        if self._mmap_store is not None:
            self._persist_cursor()
            self._mmap_store.flush()

    def add(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        if self.storage_device.type == "cpu":
            obs = _tree_to_device(obs, self.storage_device)
            next_obs = _tree_to_device(next_obs, self.storage_device)
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self._advance()
        self._persist_cursor()

    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> ReplayBufferSample:
        obs_sample = {
            k: _tree_to_device(v, self.sample_device)
            for k, v in self.obs[batch_inds, env_inds].items()
        }
        next_obs_sample = {
            k: _tree_to_device(v, self.sample_device)
            for k, v in self.next_obs[batch_inds, env_inds].items()
        }
        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )

    def sample(self, batch_size: int) -> ReplayBufferSample:
        upper = self.size
        batch_inds = torch.randint(0, upper, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return self._index_batch(batch_inds, env_inds)
