"""Generic recurrent (LSTM/GRU) latent-processing module.

Sits between a BaseFeaturesExtractor and policy/value heads. Consumes a flat
(B, D) or (T, B, D) latent tensor -- never raw observations -- so it is
generic over the *type* of upstream encoder (FlattenExtractor, PlainConv,
ResNetEncoder, ...) without any encoder-specific casing.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn as nn

RNNType = Literal["lstm", "gru"]
RecurrentState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
# GRU: Tensor of shape (num_layers, B, hidden_size).
# LSTM: tuple (h, c), each (num_layers, B, hidden_size).


# ---------------------------------------------------------------------------
# NOTE(future-work): why SAC / off-policy is NOT wired to this module (yet)
# ---------------------------------------------------------------------------
# This module is deliberately usable both for a single rollout step and for a
# full (T, B, D) BPTT window (see `step()` / `forward_sequence()` below), and
# nothing about its forward-pass API is PPO-specific. The reason only PPO
# consumes it this round is entirely on the *buffer/sampling* side, not here:
#
# PPO is on-policy: `train()` runs immediately after `RolloutBuffer.add()`
# finishes collecting a window, under (approximately) the same policy
# parameters that produced the data. That lets `RolloutBuffer.get_sequences()`
# always reconstruct BPTT windows starting from the TRUE hidden state that was
# live at collection time (captured once per rollout, masked at episode
# boundaries -- see PPO._setup_model()/train()) with zero staleness. No
# stored per-timestep hidden state, no burn-in, no padding.
#
# SAC (and any off-policy algorithm) breaks that assumption: replay sampling
# draws a WINDOW that starts at an arbitrary point in the buffer, not the true
# start of collection, and by the time it's replayed the policy/critic
# parameters have moved since that data was collected. A hidden state
# recomputed from a zero-init at the sampled window's start does NOT match
# what the CURRENT network would have produced walking in from the actual
# episode start -- it's simply the wrong conditioning state, and the
# mismatch compounds with off-policy staleness.
#
# R2D2 (Kapturowski et al., "Recurrent Experience Replay in Distributed
# Reinforcement Learning", ICLR 2019) is the reference fix, and it combines
# TWO mechanisms -- their own ablation shows either one alone underperforms:
#   1. Store the RNN hidden state produced at collection time alongside each
#      stored transition/sequence-start, so replay has a real starting point
#      instead of a zero-init.
#   2. Burn-in: before the actual loss/BPTT window, re-unroll the RNN under
#      CURRENT parameters over a preceding window of transitions (no gradient
#      through the burn-in portion) so the hidden state entering the loss
#      window is freshly computed by today's weights rather than stale from
#      collection time. Stored-state seeds the burn-in; it is not used
#      directly as the loss-window's initial state.
#
# Concretely, wiring this module into SAC (or DDPG, or any replay-based
# algorithm here) requires THREE things this round does not build:
#   (a) Replay buffer storage for a hidden state per transition or per
#       sequence-start. `rl_garden/buffers/tensor_buffer.py`,
#       `dict_buffer.py`, and `nstep_buffer.py` currently store only flat
#       per-transition fields (obs/next_obs/actions/rewards/dones) with no
#       slot for auxiliary per-timestep state.
#   (b) A sequence-aware sampler. `TensorReplayBuffer.sample()` (see
#       `tensor_buffer.py::sample()`) draws fully IID
#       `(batch_inds, env_inds)` pairs via `torch.randint(...)` for both
#       axes -- no contiguity guarantee between consecutive samples at all.
#       An R2D2-style sampler needs to draw a contiguous
#       (burn_in_len + loss_len, ...) block per sample, anchored so the
#       stored hidden state at the block's start is valid.
#   (c) A burn-in-aware unroll mode on THIS module: something like
#       `forward_sequence(latent, state, episode_starts, burn_in_len=...)`
#       that runs the first `burn_in_len` steps under `torch.no_grad()` (or
#       with gradients detached) purely to refresh `state`, then continues
#       the remaining steps normally with gradients enabled for the loss.
#       Today's `forward_sequence()` has no burn-in concept and assumes its
#       caller's `state` argument is already the correct, un-stale starting
#       point -- true for PPO's own-collection-then-immediate-update
#       pattern, not for replay.
#
# None of the above exists yet anywhere in the repo. Do not assume
# `forward_sequence()` is replay-safe just because the shapes line up.
# ---------------------------------------------------------------------------


class RecurrentLatentEncoder(nn.Module):
    """Wraps ``nn.LSTM``/``nn.GRU`` behind an explicit-state, encoder-agnostic API.

    Hidden state is always an explicit input/output, never stored on ``self`` --
    required so the module composes cleanly with checkpointing and stays a pure
    function of ``(latent, state)``, matching how the rest of this codebase's
    actor/critic heads are already stateless per-call.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        *,
        rnn_type: RNNType = "lstm",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.rnn_type: RNNType = rnn_type
        self.num_layers = num_layers

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers)

    @property
    def features_dim(self) -> int:
        return self.hidden_size

    def get_initial_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        if self.rnn_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h, c)
        return h

    @staticmethod
    def mask_state(state: RecurrentState, keep_mask: torch.Tensor) -> RecurrentState:
        """``keep_mask``: (B,) of 1.0=keep / 0.0=reset. Broadcasts over (num_layers,B,H)."""

        def _mask(h: torch.Tensor) -> torch.Tensor:
            k = keep_mask.reshape(1, -1, *([1] * (h.dim() - 2))).to(dtype=h.dtype, device=h.device)
            return h * k

        if isinstance(state, tuple):
            return tuple(_mask(h) for h in state)
        return _mask(state)

    @staticmethod
    def index_state(state: RecurrentState, indices: torch.Tensor) -> RecurrentState:
        """Index along the batch dim (dim=1). ``indices`` may be int or bool tensor."""

        def _index(h: torch.Tensor) -> torch.Tensor:
            return h[:, indices]

        if isinstance(state, tuple):
            return tuple(_index(h) for h in state)
        return _index(state)

    def step(
        self,
        latent: torch.Tensor,
        state: RecurrentState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Single rollout step. ``latent``: (B, input_dim); ``episode_starts``: (B,)."""
        if episode_starts is not None:
            state = self.mask_state(state, 1.0 - episode_starts.float())
        output, new_state = self.rnn(latent.unsqueeze(0), state)
        return output.squeeze(0), new_state

    def forward_sequence(
        self,
        latent: torch.Tensor,
        state: RecurrentState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """BPTT update call. ``latent``: (T, B, input_dim); ``episode_starts``: (T, B).

        Loops ``step()`` T times so that a mid-sequence episode boundary is handled
        via per-step masking rather than requiring the caller to hard-split the
        sequence. ``episode_starts[0]`` is honored as given -- callers that have
        already folded the window-boundary reset into ``state`` (see PPO's rollout
        loop) should pass ``episode_starts[0] == 0``.
        """
        num_steps = latent.shape[0]
        outputs = []
        for t in range(num_steps):
            step_starts = episode_starts[t] if episode_starts is not None else None
            output, state = self.step(latent[t], state, step_starts)
            outputs.append(output)
        return torch.stack(outputs, dim=0), state

    def forward(
        self,
        latent: torch.Tensor,
        state: RecurrentState,
        episode_starts: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        """Convenience ndim-dispatch wrapper: ``latent.dim()==2`` -> ``step()``,
        ``latent.dim()==3`` -> ``forward_sequence()``. Production call sites
        (PPOPolicy, PPO rollout/train loops) call ``step()``/``forward_sequence()``
        explicitly and do NOT rely on this dispatch, since a length-1 window
        ``(1,B,D)`` and a single step ``(B,D)`` are both valid but require
        differently-shaped ``episode_starts``. This exists only so the module
        still behaves like a normal ``nn.Module`` for isolated tests.
        """
        if latent.dim() == 2:
            return self.step(latent, state, episode_starts)
        if latent.dim() == 3:
            return self.forward_sequence(latent, state, episode_starts)
        raise ValueError(f"latent must have shape (B,D) or (T,B,D), got {tuple(latent.shape)}")
