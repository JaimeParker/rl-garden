"""N-step Dict replay buffer for DrQ-v2 / DDPG.

Reuses ``DictArray`` for GPU-native storage and adds episode-boundary tracking
so n-step returns can be computed correctly at sample time without leaking
across episode boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import (
    DictArray,
    _resolve_dtype,
    _tensor_to_device,
    _tree_to_device,
)
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
        self.observation_space = observation_space

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
        self.next_obs = self._build_next_obs_storage(shape, observation_space)
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

    def _build_next_obs_storage(
        self,
        shape: tuple[int, int],
        observation_space: spaces.Dict,
    ) -> Optional[DictArray]:
        return DictArray(
            shape,
            observation_space,
            device=self.storage_device,
            mmap_store=self._mmap_store,
            mmap_path=("next_obs",),
        )

    def _before_overwrite(self, pos: int) -> None:
        del pos

    def _store_next_obs(
        self,
        next_obs: dict[str, torch.Tensor],
        episode_end_bool: torch.Tensor,
    ) -> None:
        del episode_end_bool
        assert self.next_obs is not None
        self.next_obs[self.pos] = next_obs

    def _next_obs_at(self, inds, env_inds) -> dict[str, torch.Tensor]:
        assert self.next_obs is not None
        return self.next_obs[inds, env_inds]

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

        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool()
        self._before_overwrite(self.pos)

        self.obs[self.pos] = obs
        self._store_next_obs(next_obs, episode_end_bool)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
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
                return n_reward, n_discount, self._next_obs_at(idx, e)
            if bool(self.episode_ends[idx, e].item()):
                return n_reward, n_discount, self._next_obs_at(idx, e)
        next_idx = (t + self.nstep - 1) % self.per_env_buffer_size
        return n_reward, n_discount, self._next_obs_at(next_idx, e)

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

        return rewards, discounts, self._next_obs_at(next_inds, env_inds)

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


def _copy_dict_array(src: DictArray, dst: DictArray, count: int) -> None:
    for key, value in src.data.items():
        if isinstance(value, DictArray):
            _copy_dict_array(value, dst.data[key], count)
        else:
            dst.data[key][:count].copy_(value[:count])


class LazyNextNStepDictReplayBuffer(NStepDictReplayBuffer):
    """N-step dict replay that stores only sparse episode-end next observations.

    Normal bootstrap observations are reconstructed from later ``obs`` slots in
    the same environment stream. Only reset/final observations are stored in a
    compact side table, which saves the full ``next_obs`` tensor for long
    fixed-horizon visual tasks.
    """

    def __init__(
        self,
        *args,
        pin_sampled_batch: bool = False,
        final_obs_capacity: Optional[int] = None,
        mmap_dir: Optional[str | Path] = None,
        storage_device: torch.device | str = "cuda",
        **kwargs,
    ) -> None:
        if mmap_dir is not None:
            raise ValueError("lazy next_obs replay is not supported with mmap_dir")
        if torch.device(storage_device).type != "cpu":
            raise ValueError("lazy next_obs replay requires CPU storage")
        self.pin_sampled_batch = pin_sampled_batch
        self._final_obs_capacity_arg = final_obs_capacity
        super().__init__(
            *args,
            storage_device=storage_device,
            mmap_dir=mmap_dir,
            **kwargs,
        )
        initial_capacity = (
            int(final_obs_capacity)
            if final_obs_capacity is not None
            else max(1024, self.buffer_size // 64)
        )
        if initial_capacity <= 0:
            raise ValueError("final_obs_capacity must be positive")
        self._final_obs = DictArray(
            (initial_capacity,),
            self.observation_space,
            device=self.storage_device,
        )
        self._final_slot_ids = torch.full(
            (self.per_env_buffer_size, self.num_envs),
            -1,
            dtype=torch.long,
            device=self.storage_device,
        )
        self._free_final_slots: list[int] = []
        self._next_final_slot = 0

    def _build_next_obs_storage(
        self,
        shape: tuple[int, int],
        observation_space: spaces.Dict,
    ) -> Optional[DictArray]:
        del shape, observation_space
        return None

    def _before_overwrite(self, pos: int) -> None:
        old_slots = self._final_slot_ids[pos]
        for slot in old_slots[old_slots >= 0].tolist():
            self._free_final_slots.append(int(slot))
        old_slots.fill_(-1)

    def _grow_final_obs(self) -> None:
        current = self._final_obs.shape[0]
        grown = DictArray(
            (current * 2,),
            self.observation_space,
            device=self.storage_device,
        )
        _copy_dict_array(self._final_obs, grown, current)
        self._final_obs = grown

    def _allocate_final_slot(self) -> int:
        if self._free_final_slots:
            return self._free_final_slots.pop()
        if self._next_final_slot >= self._final_obs.shape[0]:
            self._grow_final_obs()
        slot = self._next_final_slot
        self._next_final_slot += 1
        return slot

    def _store_tree_leaf(
        self,
        storage: Any,
        slot: int,
        value: Any,
        env: int,
    ) -> None:
        if isinstance(storage, DictArray):
            for key in storage.data:
                self._store_tree_leaf(storage.data[key], slot, value[key], env)
        else:
            storage[slot] = value[env]

    def _store_next_obs(
        self,
        next_obs: dict[str, torch.Tensor],
        episode_end_bool: torch.Tensor,
    ) -> None:
        for env in episode_end_bool.nonzero(as_tuple=False).flatten().tolist():
            slot = self._allocate_final_slot()
            self._final_slot_ids[self.pos, env] = slot
            self._store_tree_leaf(self._final_obs, slot, next_obs, int(env))

    def _final_obs_at_slots(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._final_obs[slots]

    def _next_obs_at(self, inds, env_inds) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(inds):
            slot = int(self._final_slot_ids[int(inds), int(env_inds)].item())
            if slot >= 0:
                return self._final_obs[slot]
            next_idx = (int(inds) + 1) % self.per_env_buffer_size
            return self.obs[next_idx, int(env_inds)]

        slots = self._final_slot_ids[inds, env_inds]
        normal_next_inds = (inds + 1) % self.per_env_buffer_size
        normal_next = self.obs[normal_next_inds, env_inds]
        if not (slots >= 0).any():
            return normal_next

        result = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in normal_next.items()
        }
        final_mask = slots >= 0
        final_values = self._final_obs_at_slots(slots[final_mask])
        for key, value in result.items():
            value[final_mask] = final_values[key]
        return result

    def _valid_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> torch.Tensor:
        valid = super()._valid_nstep_batch(batch_inds, env_inds)
        target_ep = self._ep_id[batch_inds, env_inds]
        base_step = self._step_id[batch_inds, env_inds]
        active = valid.clone()

        for i in range(self.nstep):
            idx = (batch_inds + i) % self.per_env_buffer_size
            stopped = self.dones[idx, env_inds] | self.episode_ends[idx, env_inds]
            active &= ~stopped

        next_idx = (batch_inds + self.nstep) % self.per_env_buffer_size
        has_bootstrap_obs = (
            (self._ep_id[next_idx, env_inds] == target_ep)
            & (self._step_id[next_idx, env_inds] == base_step + self.nstep)
        )
        return valid & (~active | has_bootstrap_obs)

    def _valid_nstep(self, t: int, e: int) -> bool:
        if not super()._valid_nstep(t, e):
            return False
        target_ep = self._ep_id[t, e].item()
        base_step = self._step_id[t, e].item()
        for i in range(self.nstep):
            idx = (t + i) % self.per_env_buffer_size
            if bool(self.dones[idx, e].item()) or bool(
                self.episode_ends[idx, e].item()
            ):
                return True
        next_idx = (t + self.nstep) % self.per_env_buffer_size
        return (
            self._ep_id[next_idx, e].item() == target_ep
            and self._step_id[next_idx, e].item() == base_step + self.nstep
        )

    def _compute_nstep_batch(
        self,
        batch_inds: torch.Tensor,
        env_inds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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

        next_obs = self._next_obs_at(next_inds, env_inds)
        bootstrap_next = active
        if bootstrap_next.any():
            bootstrap_inds = (batch_inds[bootstrap_next] + self.nstep) % (
                self.per_env_buffer_size
            )
            bootstrap_envs = env_inds[bootstrap_next]
            bootstrap_obs = self.obs[bootstrap_inds, bootstrap_envs]
            for key, value in next_obs.items():
                value[bootstrap_next] = bootstrap_obs[key]
        return rewards, discounts, next_obs

    def sample(self, batch_size: int) -> NStepReplayBufferSample:
        upper = self.per_env_buffer_size if self.full else self.pos
        if upper < self.nstep + 1:
            raise RuntimeError(
                f"Buffer has only {upper} transitions per env but lazy "
                f"nstep={self.nstep} sampling needs at least nstep+1."
            )

        batch_inds, env_inds = self._sample_valid_indices(batch_size, upper)
        rewards, discounts, next_obs = self._compute_nstep_batch(
            batch_inds, env_inds
        )
        pin = self.pin_sampled_batch and self.sample_device.type == "cuda"
        return NStepReplayBufferSample(
            obs=_tree_to_device(
                self.obs[batch_inds, env_inds],
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
            next_obs=_tree_to_device(
                next_obs,
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
            actions=_tensor_to_device(
                self.actions[batch_inds, env_inds],
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
            rewards=_tensor_to_device(
                rewards,
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
            dones=_tensor_to_device(
                discounts == 0.0,
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
            discounts=_tensor_to_device(
                discounts,
                self.sample_device,
                non_blocking=pin,
                pin_memory=pin,
            ),
        )
