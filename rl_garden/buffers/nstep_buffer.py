"""N-step Dict replay buffer for DrQ-v2 / DDPG.

Reuses ``DictArray`` for GPU-native storage and adds episode-boundary tracking
so n-step returns can be computed correctly at sample time without leaking
across episode boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray, _resolve_dtype, _tree_to_device
from rl_garden.buffers.mmap_storage import (
    MmapMode,
    MmapTensorStore,
    space_metadata,
)
from rl_garden.common.types import NStepReplayBufferSample


class NStepDictReplayBuffer(BaseReplayBuffer):
    """Dict replay buffer with n-step return support.

    Storage layout is identical to ``DictReplayBuffer``:
    ``(per_env_buffer_size, num_envs, *shape)`` ring buffer.

    Episode boundaries are tracked via a per-transition episode-id tensor so
    n-step lookahead never crosses into a different episode.

    Parameters
    ----------
    nstep : int
        Number of steps for n-step return (default 3, matching DrQ-v2).
    gamma : float
        Discount factor for n-step reward accumulation (default 0.99).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        nstep: int = 3,
        gamma: float = 0.99,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
        mmap_dir: Optional[str | Path] = None,
        mmap_mode: MmapMode = "create",
    ) -> None:
        assert isinstance(observation_space, spaces.Dict), (
            "NStepDictReplayBuffer requires a Dict observation space."
        )
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")

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
        self.nstep = nstep
        self.gamma = gamma

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
                    "nstep": nstep,
                    "gamma": gamma,
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
            self.dones = torch.zeros(
                shape, dtype=torch.bool, device=self.storage_device
            )
            self.episode_ends = torch.zeros(
                shape, dtype=torch.bool, device=self.storage_device
            )
            self._ep_id = torch.full(
                shape, -1, dtype=torch.long, device=self.storage_device
            )
            self._current_ep_id = torch.zeros(
                num_envs, dtype=torch.long, device=self.storage_device
            )
            self._step_id = torch.full(
                shape, -1, dtype=torch.long, device=self.storage_device
            )
            self._current_step_id = torch.zeros(
                num_envs, dtype=torch.long, device=self.storage_device
            )
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
                ("dones",), shape=shape, dtype=torch.bool
            )
            self.episode_ends = self._mmap_store.tensor(
                ("episode_ends",), shape=shape, dtype=torch.bool
            )
            self._ep_id = self._mmap_store.tensor(
                ("nstep", "episode_ids"),
                shape=shape,
                dtype=torch.int64,
                fill_value=-1,
            )
            self._current_ep_id = self._mmap_store.tensor(
                ("nstep", "current_episode_ids"),
                shape=(num_envs,),
                dtype=torch.int64,
            )
            self._step_id = self._mmap_store.tensor(
                ("nstep", "step_ids"),
                shape=shape,
                dtype=torch.int64,
                fill_value=-1,
            )
            self._current_step_id = self._mmap_store.tensor(
                ("nstep", "current_step_ids"),
                shape=(num_envs,),
                dtype=torch.int64,
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

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(
        self,
        obs: dict[str, torch.Tensor],
        next_obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> None:
        if self.storage_device.type == "cpu":
            obs = _tree_to_device(obs, self.storage_device)
            next_obs = _tree_to_device(next_obs, self.storage_device)
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
            episode_end = episode_end.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool()
        self.dones[self.pos] = done_bool
        self.episode_ends[self.pos] = episode_end_bool

        # Episode boundary tracking: assign current episode id.
        self._ep_id[self.pos] = self._current_ep_id.to(self.storage_device)
        self._step_id[self.pos] = self._current_step_id.to(self.storage_device)
        # Advance episode counter whenever the environment resets, independently
        # from whether value bootstrapping should stop at that boundary.
        self._current_ep_id += episode_end_bool.long()
        self._current_step_id += 1

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
        self._persist_cursor()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _valid_nstep(self, t: int, e: int) -> bool:
        """Check that the n-step window at (t, e) is temporally contiguous."""
        target_ep = self._ep_id[t, e].item()
        base_step = self._step_id[t, e].item()
        if target_ep < 0 or base_step < 0:
            return False
        for i in range(1, self.nstep):
            prev_idx = (t + i - 1) % self.per_env_buffer_size
            if bool(self.episode_ends[prev_idx, e].item()):
                return True
            idx = (t + i) % self.per_env_buffer_size
            if self._ep_id[idx, e].item() != target_ep:
                return False
            if self._step_id[idx, e].item() != base_step + i:
                return False
        return True

    def _compute_nstep(
        self, t: int, e: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (nstep_reward, nstep_discount, next_obs) for window at (t, e)."""
        n_reward = torch.zeros_like(self.rewards[t, e])
        n_discount = torch.ones_like(self.rewards[t, e])
        for i in range(self.nstep):
            idx = (t + i) % self.per_env_buffer_size
            n_reward += n_discount * self.rewards[idx, e]
            n_discount *= self.gamma
            if bool(self.dones[idx, e].item()):
                n_discount = torch.zeros_like(n_discount)
                return n_reward, n_discount, self.next_obs[idx, e]
            if bool(self.episode_ends[idx, e].item()):
                return n_reward, n_discount, self.next_obs[idx, e]
        next_idx = (t + self.nstep - 1) % self.per_env_buffer_size
        return n_reward, n_discount, self.next_obs[next_idx, e]

    def _valid_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Return a validity mask for candidate n-step starting positions."""
        target_ep = self._ep_id[batch_inds, env_inds]
        base_step = self._step_id[batch_inds, env_inds]
        valid = (target_ep >= 0) & (base_step >= 0)
        active = valid.clone()

        for i in range(1, self.nstep):
            prev_idx = (batch_inds + i - 1) % self.per_env_buffer_size
            active &= ~self.episode_ends[prev_idx, env_inds]

            idx = (batch_inds + i) % self.per_env_buffer_size
            contiguous = (
                (self._ep_id[idx, env_inds] == target_ep)
                & (self._step_id[idx, env_inds] == base_step + i)
            )
            valid &= ~active | contiguous
            active &= contiguous

        return valid

    def _sample_valid_indices(
        self,
        batch_size: int,
        upper: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample uniformly from valid (time, env) pairs using batched rejection."""
        accepted_batch: list[torch.Tensor] = []
        accepted_env: list[torch.Tensor] = []
        remaining = batch_size
        attempted = 0
        max_attempts = max(1_000, batch_size * 100) * batch_size
        index_device = self.storage_device

        while remaining > 0:
            candidate_count = max(32, remaining * 2)
            env_inds = torch.randint(
                0, self.num_envs, (candidate_count,), device=index_device
            )
            batch_inds = torch.randint(
                0, upper, (candidate_count,), device=index_device
            )
            valid = self._valid_nstep_batch(batch_inds, env_inds)
            if valid.any():
                valid_batch = batch_inds[valid][:remaining]
                valid_env = env_inds[valid][:remaining]
                accepted_batch.append(valid_batch)
                accepted_env.append(valid_env)
                remaining -= valid_batch.numel()

            attempted += candidate_count
            if attempted >= max_attempts and remaining > 0:
                raise RuntimeError(
                    "Could not sample enough valid n-step windows. The buffer may "
                    "contain too few temporally contiguous transitions."
                )

        return torch.cat(accepted_batch), torch.cat(accepted_env)

    def _compute_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute n-step returns for a batch of valid starting positions."""
        rewards = torch.zeros_like(self.rewards[batch_inds, env_inds])
        discounts = torch.ones_like(rewards)
        active = torch.ones_like(rewards, dtype=torch.bool)
        next_inds = (batch_inds + self.nstep - 1) % self.per_env_buffer_size

        for i in range(self.nstep):
            idx = (batch_inds + i) % self.per_env_buffer_size
            step_rewards = self.rewards[idx, env_inds]
            rewards += torch.where(active, discounts * step_rewards, 0.0)

            discounts = torch.where(active, discounts * self.gamma, discounts)
            terminal = active & self.dones[idx, env_inds]
            episode_end = active & self.episode_ends[idx, env_inds]
            stopped = terminal | episode_end
            discounts = torch.where(terminal, 0.0, discounts)
            next_inds = torch.where(stopped, idx, next_inds)
            active &= ~stopped

        return rewards, discounts, self.next_obs[next_inds, env_inds]

    def sample(self, batch_size: int) -> NStepReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
        if upper < self.nstep:
            raise RuntimeError(
                f"Buffer has only {upper} transitions per env but nstep={self.nstep}. "
                "Wait for more data before sampling."
            )

        batch_inds, env_inds = self._sample_valid_indices(batch_size, upper)
        rewards, discounts, next_obs = self._compute_nstep_batch(
            batch_inds, env_inds
        )

        return NStepReplayBufferSample(
            obs=_tree_to_device(
                self.obs[batch_inds, env_inds], self.sample_device
            ),
            next_obs=_tree_to_device(next_obs, self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=rewards.to(self.sample_device),
            dones=(discounts == 0.0).to(self.sample_device),
            discounts=discounts.to(self.sample_device),
        )
