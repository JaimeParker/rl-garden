"""Mmap-backed multitask episode buffer for TD-MPC2's offline multitask
training, sized to hold an entire dataset (upstream's mt80 config allocates
``buffer_size=550_450_000`` transitions -- far larger than fits in RAM/VRAM as
dense tensors, which is why upstream uses torchrl's ``LazyTensorStorage``).
Backed by ``rl_garden.buffers.mmap_storage.MmapTensorStore`` instead (already
an optional backend for ``NStepDictReplayBuffer`` -- reused, not new).

Shares ``EpisodeSliceSamplingMixin`` with ``EpisodeSliceBuffer``: identical
episode-boundary-strict windowed sampling, just over mmap-backed storage with
an extra ``task`` field. Unlike the online single-task buffer, this one is
populated exactly once via bulk ``load_episode()`` calls (no online rollout,
no ring-buffer overwrite) -- ``add()`` is therefore unsupported.

``_step_id`` here is reset to 0 at the start of every episode (whereas the
online buffer's is a monotonic counter running across the whole buffer
lifetime) rather than a global monotonic counter: since ``load_episode``
refuses to overwrite already-written positions (raises if the buffer is too
small), there is no ring-buffer wraparound to detect staleness against, so
only *within-episode* contiguity matters, and the sampling mixin's contiguity
check never compares ``_step_id`` across two different episodes anyway (an
``_ep_id`` mismatch always rejects those candidates first).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rl_garden.buffers._episode_slice_sampling import EpisodeSliceSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.mmap_storage import MmapMode, MmapTensorStore


@dataclass
class MultitaskEpisodeBufferSample:
    obs: torch.Tensor       # (horizon+1, B, obs_dim) -- zero-padded to max(obs_dims)
    action: torch.Tensor    # (horizon, B, action_dim) -- zero-padded to max(action_dims)
    reward: torch.Tensor    # (horizon, B)
    task: torch.Tensor      # (B,) -- one task id per sampled window (constant within a window)


class MmapMultitaskEpisodeBuffer(EpisodeSliceSamplingMixin, BaseReplayBuffer):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        buffer_size: int,
        horizon: int,
        mmap_dir: str | Path,
        mmap_mode: MmapMode = "create",
        storage_device: torch.device | str = "cpu",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        storage_device = torch.device(storage_device)
        if storage_device.type != "cpu":
            raise ValueError("MmapMultitaskEpisodeBuffer requires CPU storage_device.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = 1
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size
        if self.per_env_buffer_size <= horizon:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be > horizon ({horizon})."
            )
        self.horizon = horizon
        self.storage_device = storage_device
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False

        self._store = MmapTensorStore(
            mmap_dir,
            mode=mmap_mode,
            manifest={
                "buffer_class": "MmapMultitaskEpisodeBuffer",
                "buffer_size": buffer_size,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "horizon": horizon,
            },
        )
        shape = (self.per_env_buffer_size, self.num_envs)
        self.obs = self._store.tensor(("obs",), shape=shape + (obs_dim,), dtype=torch.float32)
        self.actions = self._store.tensor(("actions",), shape=shape + (action_dim,), dtype=torch.float32)
        self.rewards = self._store.tensor(("rewards",), shape=shape, dtype=torch.float32)
        self.dones = self._store.tensor(("dones",), shape=shape, dtype=torch.bool)
        self.episode_ends = self._store.tensor(("episode_ends",), shape=shape, dtype=torch.bool)
        self.task = self._store.tensor(("task",), shape=shape, dtype=torch.int64, fill_value=-1)
        self._ep_id = self._store.tensor(("ep_id",), shape=shape, dtype=torch.int64, fill_value=-1)
        self._step_id = self._store.tensor(("step_id",), shape=shape, dtype=torch.int64, fill_value=-1)

        self._current_ep_id = torch.zeros(self.num_envs, dtype=torch.long)

    def flush(self) -> None:
        self._store.flush()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "MmapMultitaskEpisodeBuffer is populated once via load_episode() "
            "from a converted offline dataset; it does not support online "
            "single-step add()."
        )

    def load_episode(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        task_idx: int,
    ) -> None:
        """Bulk-write one episode's worth of already-padded transitions.

        ``obs``/``action``/``reward`` all have length ``L`` (the episode's
        transition count) -- matching every other buffer in this package,
        the true final observation (the ``L``-th, after the last action) is
        not stored; see ``episode_slice_buffer.py``'s module docstring for
        why a sampled window can never reach it anyway.
        """
        length = obs.shape[0]
        if length < 1:
            raise ValueError("episode must have at least 1 step, got 0.")
        if self.pos + length > self.per_env_buffer_size:
            raise RuntimeError(
                "MmapMultitaskEpisodeBuffer is too small to hold this episode: "
                f"pos={self.pos}, episode_len={length}, "
                f"per_env_buffer_size={self.per_env_buffer_size}."
            )

        idx = slice(self.pos, self.pos + length)
        self.obs[idx, 0] = obs.to(torch.float32)
        self.actions[idx, 0] = action.to(torch.float32)
        self.rewards[idx, 0] = reward.to(torch.float32)
        self.dones[idx, 0] = False
        episode_end = torch.zeros(length, dtype=torch.bool)
        episode_end[-1] = True
        self.episode_ends[idx, 0] = episode_end
        self.task[idx, 0] = task_idx
        self._step_id[idx, 0] = torch.arange(length, dtype=torch.long)
        self._ep_id[idx, 0] = int(self._current_ep_id[0].item())

        self.pos += length
        self._current_ep_id[0] += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> MultitaskEpisodeBufferSample:
        t0, env_inds = self._sample_valid_window_starts(batch_size)
        idx_grid, env_grid = self._gather_window(t0, env_inds)

        obs = self.obs[idx_grid, env_grid]
        action = self.actions[idx_grid[:-1], env_grid[:-1]]
        reward = self.rewards[idx_grid[:-1], env_grid[:-1]]
        task = self.task[t0, env_inds]

        return MultitaskEpisodeBufferSample(
            obs=obs.to(self.sample_device),
            action=action.to(self.sample_device),
            reward=reward.to(self.sample_device),
            task=task.to(self.sample_device),
        )
