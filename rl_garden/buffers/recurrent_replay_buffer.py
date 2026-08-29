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

from rl_garden.buffers._checkpointed_sequence_buffer import _CheckpointedSequenceReplayBuffer
from rl_garden.buffers.dict_buffer import _tree_to_device
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


class RecurrentReplayBuffer(_CheckpointedSequenceReplayBuffer):
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
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self._has_cell_state = rnn_type == "lstm"
        super().__init__(
            observation_space,
            action_space,
            num_envs,
            buffer_size,
            burn_in_len=burn_in_len,
            learning_len=learning_len,
            forward_len=forward_len,
            stride=burn_in_len,
            gamma=gamma,
            prio_exponent=prio_exponent,
            importance_sampling_exponent=importance_sampling_exponent,
            priority_eps=priority_eps,
            storage_device=storage_device,
            sample_device=sample_device,
        )

    def _init_checkpoint_extra_storage(self) -> None:
        self.hidden_checkpoints_h = torch.zeros(
            self.checkpoint_capacity,
            self.num_envs,
            self.rnn_num_layers,
            self.rnn_hidden_size,
            device=self.storage_device,
        )
        self.hidden_checkpoints_c = (
            torch.zeros(
                self.checkpoint_capacity,
                self.num_envs,
                self.rnn_num_layers,
                self.rnn_hidden_size,
                device=self.storage_device,
            )
            if self._has_cell_state
            else None
        )
        self.hidden_checkpoint_ep_id = torch.full(
            (self.checkpoint_capacity, self.num_envs),
            -1,
            dtype=torch.long,
            device=self.storage_device,
        )

    def _write_checkpoint_extra(
        self, slots: torch.Tensor, envs: torch.Tensor, hidden: RecurrentState = None
    ) -> None:
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
        episode_end_bool = self._add_common(obs, next_obs, action, reward, done, episode_end)
        self._write_checkpoint_slots(episode_end_bool, hidden=hidden)

        self._current_ep_id = self._current_ep_id + episode_end_bool.long()
        self._current_step_id = self._current_step_id + 1
        self._current_ep_relative_step = torch.where(
            episode_end_bool,
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
        (
            env_inds,
            leaf_indices,
            is_weights,
            window_obs,
            actions,
            episode_starts,
            rewards,
            discounts,
        ) = self._sample_common(batch_size, generator=generator)

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
