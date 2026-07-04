"""Off-policy replay buffer for recurrent (RNN, later Transformer) latent modules.

Ports R2D2's core mechanism (Kapturowski et al. 2019: stored hidden state +
burn-in + n-step + priority replay) into rl-garden's ring-buffer architecture,
reimplemented in this codebase's vectorized (torch tensor ops, no Python
per-sample loops) idiom rather than the reference clone's Python/numpy style --
see ``rl_garden/networks/recurrent.py``'s module docstring for the off-policy
staleness problem this solves.

Checkpoint-aligned sampling grid: the hidden state is stored once per ``stride
== burn_in_len`` steps (measured from each episode's own start, not the buffer's
absolute position), and every sampled window's start ``t0`` is constrained to a
checkpoint position. This means burn-in length is always exactly ``burn_in_len``
(seeded from the checkpoint stored AT ``t0``, which is either a genuine
collection-time state or -- for an episode's very first checkpoint -- the
all-zero initial state) with no per-sample variable-length bookkeeping needed;
``episode_starts`` correctly reflects any episode boundary anywhere in the
window, including inside burn-in itself, via the existing ``mask_state``
mechanism in ``RecurrentLatentEncoder``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from gymnasium import spaces

from rl_garden.buffers._recurrent_sampling import RecurrentSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray, _tree_to_device
from rl_garden.buffers.sum_tree import SumTree
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs

# Local, intentionally-duplicated type alias mirroring
# rl_garden.networks.recurrent.RecurrentState -- rl_garden/buffers/ has zero
# dependency on rl_garden/networks/ today (see recurrent_rollout_buffer.py's
# identical precedent for this exact duplication).
RecurrentState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


@dataclass
class RecurrentReplayBufferSample:
    obs: Obs                                   # (window_len, B, *obs_shape)
    # Always None -- this buffer has no separate next_obs concept (the window
    # itself covers "next" positions internally); present only so
    # SACCore.train()'s unconditional features_extractor.prepare_batch(obs,
    # next_obs) call has something to pass without a special case there.
    next_obs: Optional[Obs]
    actions: torch.Tensor                       # (learning_len, B, act_dim)
    rewards: torch.Tensor                       # (learning_len, B) -- pre-accumulated n-step
    discounts: torch.Tensor                     # (learning_len, B)
    episode_starts: torch.Tensor                # (window_len, B)
    initial_hidden_h: torch.Tensor              # (B, num_layers, H) -- batch-dim-first
    initial_hidden_c: Optional[torch.Tensor]    # (B, num_layers, H), None for GRU
    priority_indices: torch.Tensor              # (B,) LongTensor, flat leaf indices
    is_weights: torch.Tensor                    # (B,)


def _copy_tree(src, dst, count: int) -> None:
    if isinstance(src, DictArray):
        for key, value in src.data.items():
            _copy_tree(value, dst.data[key], count)
    else:
        dst[:count].copy_(src[:count])


class RecurrentReplayBuffer(RecurrentSamplingMixin, BaseReplayBuffer):
    """One class for both Box and Dict observations (unlike the Tensor/Dict
    n-step buffer pair) -- the sum-tree/checkpoint/burn-in machinery here is
    already the novel bulk of this file; duplicating it across a second class
    would double the review surface for no behavioral benefit."""

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
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
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
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs

        self.burn_in_len = burn_in_len
        self.learning_len = learning_len
        self.forward_len = forward_len
        self.stride = burn_in_len
        if self.per_env_buffer_size % self.stride != 0:
            raise ValueError(
                f"per_env_buffer_size ({self.per_env_buffer_size}) must be divisible "
                f"by burn_in_len/checkpoint stride ({self.stride})."
            )
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self._has_cell_state = rnn_type == "lstm"
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
                "RecurrentReplayBuffer supports Box or Dict observations, got "
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
        # New: steps since episode start (resets to 0 the step after an episode
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
        self.hidden_checkpoints_h = torch.zeros(
            self.checkpoint_capacity,
            num_envs,
            rnn_num_layers,
            rnn_hidden_size,
            device=self.storage_device,
        )
        self.hidden_checkpoints_c = (
            torch.zeros(
                self.checkpoint_capacity,
                num_envs,
                rnn_num_layers,
                rnn_hidden_size,
                device=self.storage_device,
            )
            if self._has_cell_state
            else None
        )
        self.hidden_checkpoint_ep_id = torch.full(
            (self.checkpoint_capacity, num_envs),
            -1,
            dtype=torch.long,
            device=self.storage_device,
        )

        # Compact final/terminal-observation side table -- gymnasium autoreset
        # means the ring buffer's naturally-following obs after an episode-end
        # position is already the NEXT episode's reset obs, not the true final
        # one. Mirrors LazyNextNStepDictReplayBuffer's exact mechanism
        # (nstep_buffer.py), generalized to Box observations too.
        self._final_slot_ids = torch.full(
            shape, -1, dtype=torch.long, device=self.storage_device
        )
        self._free_final_slots: list[int] = []
        self._next_final_slot = 0
        final_obs_capacity = max(1024, self.buffer_size // 64)
        if self._is_dict_obs:
            self._final_obs = DictArray(
                (final_obs_capacity,), observation_space, device=self.storage_device
            )
        else:
            self._final_obs = torch.zeros(
                (final_obs_capacity,) + tuple(observation_space.shape),
                device=self.storage_device,
            )

        capacity_total = self.checkpoint_capacity * num_envs
        self._priority_tree = SumTree(
            capacity=capacity_total,
            alpha=prio_exponent,
            beta=importance_sampling_exponent,
            device=self.storage_device,
            eps=priority_eps,
        )

    # ------------------------------------------------------------------
    # Final-obs side table
    # ------------------------------------------------------------------

    def _grow_final_obs(self) -> None:
        current = self._final_obs.shape[0]
        new_capacity = current * 2
        if self._is_dict_obs:
            grown = DictArray(
                (new_capacity,), self.observation_space, device=self.storage_device
            )
            _copy_tree(self._final_obs, grown, current)
        else:
            grown = torch.zeros(
                (new_capacity,) + tuple(self.observation_space.shape),
                device=self.storage_device,
            )
            grown[:current] = self._final_obs
        self._final_obs = grown

    def _allocate_final_slot(self) -> int:
        if self._free_final_slots:
            return self._free_final_slots.pop()
        if self._next_final_slot >= self._final_obs.shape[0]:
            self._grow_final_obs()
        slot = self._next_final_slot
        self._next_final_slot += 1
        return slot

    def _write_final_obs_slot(self, storage, slot: int, value, env: int) -> None:
        if isinstance(storage, DictArray):
            for key in storage.data:
                self._write_final_obs_slot(storage.data[key], slot, value[key], env)
        else:
            storage[slot] = value[env].to(self.storage_device)

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
        hidden: RecurrentState,
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

        is_checkpoint = self._current_ep_relative_step % self.stride == 0
        if is_checkpoint.any():
            envs = is_checkpoint.nonzero(as_tuple=False).flatten()
            slots = self._current_ckpt_pos[envs] % self.checkpoint_capacity
            if self._has_cell_state:
                h, c = hidden
            else:
                h, c = hidden, None
            self.hidden_checkpoints_h[slots, envs] = (
                h[:, envs].transpose(0, 1).to(self.storage_device)
            )
            if self._has_cell_state:
                self.hidden_checkpoints_c[slots, envs] = (
                    c[:, envs].transpose(0, 1).to(self.storage_device)
                )
            self.hidden_checkpoint_ep_id[slots, envs] = self._current_ep_id[envs]
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
    ) -> RecurrentReplayBufferSample:
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

        slots = leaf_indices % self.checkpoint_capacity
        initial_hidden_h = self.hidden_checkpoints_h[slots, env_inds]
        initial_hidden_c = (
            self.hidden_checkpoints_c[slots, env_inds] if self._has_cell_state else None
        )

        return RecurrentReplayBufferSample(
            obs=_tree_to_device(window_obs, self.sample_device),
            next_obs=None,
            actions=actions.to(self.sample_device),
            rewards=rewards.to(self.sample_device),
            discounts=discounts.to(self.sample_device),
            episode_starts=episode_starts.to(self.sample_device),
            initial_hidden_h=initial_hidden_h.to(self.sample_device),
            initial_hidden_c=(
                initial_hidden_c.to(self.sample_device)
                if initial_hidden_c is not None
                else None
            ),
            priority_indices=leaf_indices.to(self.sample_device),
            is_weights=is_weights.to(self.sample_device),
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor) -> None:
        self._priority_tree.update(
            indices.to(self.storage_device), td_errors.to(self.storage_device)
        )
