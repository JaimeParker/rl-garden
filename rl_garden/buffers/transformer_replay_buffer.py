"""Off-policy replay buffer for GTrXL (Transformer) latent modules.

Sibling of ``RecurrentReplayBuffer`` for a fundamentally simpler case: GTrXL's
memory is a bounded sliding window of raw activations, not a compressed
RNN-style state, so ``TransformerSAC`` never needs a stored per-transition
hidden-state checkpoint -- burn-in always starts from a fresh zero state (see
``rl_garden.networks.gtrxl.GTrXLLatentEncoder.forward_sequence_with_burn_in``
and ``TransformerSAC._initial_state_from_sample``). That removes the entire
reason ``RecurrentReplayBuffer`` constrains sampled window starts to a sparse
``stride == burn_in_len`` checkpoint grid (that grid exists only to make
storing a hidden-state snapshot at every position affordable). Here ``stride``
is simply ``1``: every buffer position is a valid, independently-sampleable
window start once it has ``burn_in_len + learning_len + forward_len`` steps of
contiguous history ahead of it. This buffer therefore reuses
``RecurrentSamplingMixin`` (sampling/contiguity/n-step accumulation) and
``SumTree`` (priority tree) completely unmodified -- both are already generic
over what "checkpoint" means -- and ``FinalObsTableMixin`` for the compact
final-obs side table, but carries none of ``RecurrentReplayBuffer``'s
hidden-state storage or shape parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._final_obs_table import FinalObsTableMixin
from rl_garden.buffers._recurrent_sampling import RecurrentSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray, _tree_to_device
from rl_garden.buffers.sum_tree import SumTree
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


@dataclass
class TransformerReplayBufferSample:
    obs: Obs                                   # (window_len, B, *obs_shape)
    # Always None -- see RecurrentReplayBufferSample's identical field for why.
    next_obs: Optional[Obs]
    actions: torch.Tensor                       # (learning_len, B, act_dim)
    rewards: torch.Tensor                       # (learning_len, B) -- pre-accumulated n-step
    discounts: torch.Tensor                     # (learning_len, B)
    episode_starts: torch.Tensor                # (window_len, B)
    priority_indices: torch.Tensor              # (B,) LongTensor, flat leaf indices
    is_weights: torch.Tensor                    # (B,)


class TransformerReplayBuffer(RecurrentSamplingMixin, FinalObsTableMixin, BaseReplayBuffer):
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
        # Every position is a valid window start (no stored hidden state to
        # align to) -- unlike RecurrentReplayBuffer, stride is not
        # burn_in_len-periodic. See module docstring.
        self.stride = 1
        self.gamma = gamma

        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False

        shape = (self.per_env_buffer_size, num_envs)
        self._is_dict_obs = isinstance(observation_space, spaces.Dict)
        if self._is_dict_obs:
            self.obs = DictArray(shape, observation_space, device=self.storage_device)
        elif isinstance(observation_space, spaces.Box):
            self.obs = torch.zeros(
                shape + tuple(observation_space.shape), device=self.storage_device
            )
        else:
            raise TypeError(
                "TransformerReplayBuffer supports Box or Dict observations, got "
                f"{type(observation_space)}."
            )
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), device=self.storage_device
        )
        self.rewards = torch.zeros(shape, device=self.storage_device)
        self.dones = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)
        self.episode_ends = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

        # Episode-contiguity bookkeeping -- same fields/semantics as
        # RecurrentReplayBuffer/NStepDictReplayBuffer, reused not reinvented.
        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )
        self._ep_relative_step = torch.full(
            shape, -1, dtype=torch.long, device=self.storage_device
        )
        self._current_ep_relative_step = torch.zeros(
            num_envs, dtype=torch.long, device=self.storage_device
        )

        # RecurrentSamplingMixin needs a slot->position table and a capacity,
        # but with stride=1 there is no compression: every position is its own
        # slot (checkpoint_capacity == per_env_buffer_size).
        self.checkpoint_capacity = self.per_env_buffer_size
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

        capacity_total = self.checkpoint_capacity * num_envs
        self._priority_tree = SumTree(
            capacity=capacity_total,
            alpha=prio_exponent,
            beta=importance_sampling_exponent,
            device=self.storage_device,
            eps=priority_eps,
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> None:
        done_bool = done.to(self.storage_device).bool()
        episode_end_bool = episode_end.to(self.storage_device).bool()

        if self._is_dict_obs:
            assert isinstance(obs, dict)
            self.obs[self.pos] = {k: v.to(self.storage_device) for k, v in obs.items()}
        else:
            assert isinstance(obs, torch.Tensor)
            self.obs[self.pos] = obs.to(self.storage_device)
        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.reshape(self.num_envs).to(self.storage_device)
        self.dones[self.pos] = done_bool.reshape(self.num_envs)
        self.episode_ends[self.pos] = episode_end_bool.reshape(self.num_envs)

        # Free any final-obs slot the position we're about to overwrite owned.
        old_slots = self._final_slot_ids[self.pos]
        for slot in old_slots[old_slots >= 0].tolist():
            self._free_final_slots.append(int(slot))
        old_slots.fill_(-1)
        for env in episode_end_bool.reshape(self.num_envs).nonzero(as_tuple=False).flatten().tolist():
            slot = self._allocate_final_slot()
            self._final_slot_ids[self.pos, env] = slot
            self._write_final_obs_slot(self._final_obs, slot, next_obs, env)

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id
        self._ep_relative_step[self.pos] = self._current_ep_relative_step

        # stride == 1: every env's every step is a "checkpoint" (a candidate
        # window start).
        envs = torch.arange(self.num_envs, device=self.storage_device)
        slots = self._current_ckpt_pos[envs] % self.checkpoint_capacity
        self._ckpt_slot_to_pos[slots, envs] = self.pos
        leaf_indices = envs * self.checkpoint_capacity + slots
        self._priority_tree.set_uninitialized(leaf_indices)
        self._current_ckpt_pos[envs] += 1

        self._current_ep_id = self._current_ep_id + episode_end_bool.reshape(self.num_envs).long()
        self._current_step_id = self._current_step_id + 1
        self._current_ep_relative_step = torch.where(
            episode_end_bool.reshape(self.num_envs),
            torch.zeros_like(self._current_ep_relative_step),
            self._current_ep_relative_step + 1,
        )

        self._advance()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> TransformerReplayBufferSample:
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

        return TransformerReplayBufferSample(
            obs=_tree_to_device(window_obs, self.sample_device),
            next_obs=None,
            actions=actions.to(self.sample_device),
            rewards=rewards.to(self.sample_device),
            discounts=discounts.to(self.sample_device),
            episode_starts=episode_starts.to(self.sample_device),
            priority_indices=leaf_indices.to(self.sample_device),
            is_weights=is_weights.to(self.sample_device),
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        self._priority_tree.update(
            indices.to(self.storage_device), td_errors.to(self.storage_device)
        )
