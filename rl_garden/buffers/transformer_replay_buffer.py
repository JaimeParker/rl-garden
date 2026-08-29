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
contiguous history ahead of it. This buffer therefore shares
``_CheckpointedSequenceReplayBuffer`` (storage/checkpoint scaffolding,
``RecurrentSamplingMixin`` sampling/contiguity/n-step accumulation,
``FinalObsTableMixin`` final-obs side table, ``SumTree`` priority tree) with
``RecurrentReplayBuffer`` unmodified, and leaves the hidden-state checkpoint
hooks at their no-op defaults -- it carries none of ``RecurrentReplayBuffer``'s
hidden-state storage or shape parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._checkpointed_sequence_buffer import _CheckpointedSequenceReplayBuffer
from rl_garden.buffers.dict_buffer import _tree_to_device
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


class TransformerReplayBuffer(_CheckpointedSequenceReplayBuffer):
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
        super().__init__(
            observation_space,
            action_space,
            num_envs,
            buffer_size,
            burn_in_len=burn_in_len,
            learning_len=learning_len,
            forward_len=forward_len,
            # Every position is a valid window start (no stored hidden state to
            # align to) -- unlike RecurrentReplayBuffer, stride is not
            # burn_in_len-periodic. See module docstring.
            stride=1,
            gamma=gamma,
            prio_exponent=prio_exponent,
            importance_sampling_exponent=importance_sampling_exponent,
            priority_eps=priority_eps,
            storage_device=storage_device,
            sample_device=sample_device,
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
        episode_end_bool = self._add_common(obs, next_obs, action, reward, done, episode_end)
        self._write_checkpoint_slots(episode_end_bool)

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
    ) -> TransformerReplayBufferSample:
        (
            _env_inds,
            leaf_indices,
            is_weights,
            window_obs,
            actions,
            episode_starts,
            rewards,
            discounts,
        ) = self._sample_common(batch_size, generator=generator)

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
