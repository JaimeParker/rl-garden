"""Shared storage/checkpoint scaffolding for sequence-aware replay buffers
(``RecurrentReplayBuffer``, ``TransformerReplayBuffer``).

Both buffers store a ring of transitions plus a sparse "checkpoint" side table
keyed by a `stride`-periodic grid (``stride == burn_in_len`` for
``RecurrentReplayBuffer``'s stored-hidden-state case, ``stride == 1`` for
``TransformerReplayBuffer``'s fresh-burn-in case -- see
``transformer_replay_buffer.py``'s module docstring). Everything except the
checkpoint slot's *extra* per-algorithm payload (RNN hidden state, for
``RecurrentReplayBuffer`` only) is identical between the two, so it lives
here; the two no-op hooks (``_init_checkpoint_extra_storage``,
``_write_checkpoint_extra``) are where that one difference plugs in.
"""
from __future__ import annotations

from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._final_obs_table import FinalObsTableMixin
from rl_garden.buffers._recurrent_sampling import RecurrentSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray
from rl_garden.buffers.sum_tree import SumTree
from rl_garden.common.obs_utils import index_obs


class _CheckpointedSequenceReplayBuffer(
    RecurrentSamplingMixin, FinalObsTableMixin, BaseReplayBuffer
):
    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        *,
        burn_in_len: int = 40,
        learning_len: int = 40,
        forward_len: int = 5,
        stride: int,
        gamma: float = 0.99,
        prio_exponent: float = 0.9,
        importance_sampling_exponent: float = 0.6,
        priority_eps: float = 1e-3,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        if burn_in_len < 1:
            raise ValueError(f"burn_in_len must be >= 1, got {burn_in_len}")
        if learning_len < 1:
            raise ValueError(f"learning_len must be >= 1, got {learning_len}")
        if forward_len < 1:
            raise ValueError(f"forward_len must be >= 1, got {forward_len}")

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs

        self.burn_in_len = burn_in_len
        self.learning_len = learning_len
        self.forward_len = forward_len
        self.stride = stride
        if self.per_env_buffer_size % self.stride != 0:
            raise ValueError(
                f"per_env_buffer_size ({self.per_env_buffer_size}) must be divisible "
                f"by the checkpoint stride ({self.stride})."
            )
        self.gamma = gamma

        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False

        shape = (self.per_env_buffer_size, num_envs)
        self._is_dict_obs = isinstance(observation_space, spaces.Dict)
        class_name = type(self).__name__
        if self._is_dict_obs:
            self.obs = DictArray(shape, observation_space, device=self.storage_device)
        elif isinstance(observation_space, spaces.Box):
            self.obs = torch.zeros(
                shape + tuple(observation_space.shape), device=self.storage_device
            )
        else:
            raise TypeError(
                f"{class_name} supports Box or Dict observations, got "
                f"{type(observation_space)}."
            )
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), device=self.storage_device
        )
        self.rewards = torch.zeros(shape, device=self.storage_device)
        self.dones = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)
        self.episode_ends = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

        # Episode-contiguity bookkeeping -- same fields/semantics as
        # NStepDictReplayBuffer (nstep_buffer.py), reused not reinvented.
        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )
        # Steps since episode start (resets to 0 the step after an episode
        # ends), used for checkpoint eligibility and episode_starts computation.
        self._ep_relative_step = torch.full(
            shape, -1, dtype=torch.long, device=self.storage_device
        )
        self._current_ep_relative_step = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )

        # Compact checkpoint side-buffer (stride-x smaller than per-timestep
        # storage) -- this is what actually saves memory, not skip-writing into
        # an equally-sized tensor.
        self.checkpoint_capacity = self.per_env_buffer_size // self.stride
        self._current_ckpt_pos = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )
        self._ckpt_slot_to_pos = torch.full(
            (self.checkpoint_capacity, num_envs),
            -1,
            dtype=torch.long,
            device=self.storage_device,
        )

        self._init_final_obs_table(shape)
        self._init_checkpoint_extra_storage()

        capacity_total = self.checkpoint_capacity * num_envs
        self._priority_tree = SumTree(
            capacity=capacity_total,
            alpha=prio_exponent,
            beta=importance_sampling_exponent,
            device=self.storage_device,
            eps=priority_eps,
        )

    def _init_checkpoint_extra_storage(self) -> None:
        """No-op default. ``RecurrentReplayBuffer`` overrides to allocate RNN
        hidden-state checkpoint tensors."""
        return

    def _write_checkpoint_extra(
        self, slots: torch.Tensor, envs: torch.Tensor, hidden=None
    ) -> None:
        """No-op default. ``RecurrentReplayBuffer`` overrides to write hidden
        state for the given checkpoint slots."""
        return

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _write_checkpoint_slots(
        self, episode_end_bool: torch.Tensor, hidden=None
    ) -> None:
        is_checkpoint = self._current_ep_relative_step % self.stride == 0
        if not is_checkpoint.any():
            return
        envs = is_checkpoint.nonzero(as_tuple=False).flatten()
        slots = self._current_ckpt_pos[envs] % self.checkpoint_capacity
        self._write_checkpoint_extra(slots, envs, hidden=hidden)
        self._ckpt_slot_to_pos[slots, envs] = self.pos
        leaf_indices = envs * self.checkpoint_capacity + slots
        self._priority_tree.set_uninitialized(leaf_indices)
        self._current_ckpt_pos[envs] += 1

    def _add_common(
        self,
        obs,
        next_obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> torch.Tensor:
        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool().reshape(self.num_envs)

        if self._is_dict_obs:
            assert isinstance(obs, dict)
            self.obs[self.pos] = {k: v.to(self.storage_device) for k, v in obs.items()}
        else:
            assert isinstance(obs, torch.Tensor)
            self.obs[self.pos] = obs.to(self.storage_device)
        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.reshape(self.num_envs).to(self.storage_device)
        self.dones[self.pos] = done_bool.reshape(self.num_envs)
        self.episode_ends[self.pos] = episode_end_bool

        self._free_final_slot(self.pos)
        self._store_final_obs_for_episode_ends(self.pos, episode_end_bool, next_obs)

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id
        self._ep_relative_step[self.pos] = self._current_ep_relative_step

        return episode_end_bool

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_common(self, batch_size: int, generator: Optional[torch.Generator] = None):
        t0, env_inds, leaf_indices, is_weights = self._sample_valid_window_starts(
            batch_size, generator=generator
        )
        idx_grid, env_grid = self._gather_window(t0, env_inds)

        window_obs = index_obs(self.obs, (idx_grid, env_grid))
        window_obs = self._patch_final_obs(window_obs, idx_grid, env_grid)

        learn_slice = slice(self.burn_in_len, self.burn_in_len + self.learning_len)
        actions = self.actions[idx_grid[learn_slice], env_grid[learn_slice]]

        ep_rel = self._ep_relative_step[idx_grid, env_grid]
        episode_starts = (ep_rel == 0).float()

        rewards, discounts = self._accumulate_nstep_window(t0, env_inds)

        return (
            env_inds,
            leaf_indices,
            is_weights,
            window_obs,
            actions,
            episode_starts,
            rewards,
            discounts,
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        self._priority_tree.update(
            indices.to(self.storage_device), td_errors.to(self.storage_device)
        )
