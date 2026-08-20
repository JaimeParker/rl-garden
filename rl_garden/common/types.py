"""Shared type aliases and data containers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import torch

TensorDict = Dict[str, torch.Tensor]
Obs = Union[torch.Tensor, TensorDict]
Schedule = Callable[[float], float]  # progress_remaining in [0, 1] -> lr


@dataclass
class ReplayBufferSample:
    """A single mini-batch drawn from a replay buffer.

    ``obs`` and ``next_obs`` are ``torch.Tensor`` for flat observations or
    ``Dict[str, torch.Tensor]`` for dict observations (e.g. RGBD + state).
    All fields live on the sample device.
    """

    obs: Obs
    next_obs: Obs
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass
class NStepReplayBufferSample(ReplayBufferSample):
    """Replay buffer sample with pre-accumulated n-step discounts."""

    discounts: torch.Tensor


@dataclass
class MCReplayBufferSample(ReplayBufferSample):
    """Replay buffer sample with Monte Carlo returns for Cal-QL."""

    mc_returns: Optional[torch.Tensor] = None


@dataclass
class ReBRACReplayBufferSample(ReplayBufferSample):
    """Replay buffer sample with the dataset's next-step action, for
    ReBRAC's critic-side BC penalty (``rebrac.py:498``, ``next_actions``).

    ``next_actions[i]`` is the action *actually taken* at ``next_obs[i]`` in
    the offline trajectory, not the current sample's own ``actions[i]``.
    Read via a plain index-shift on the underlying buffer -- can land in the
    following episode at a true terminal, matching CORL's own reference
    (``qlearning_dataset()``), which reads ``dataset["actions"][i+1]``
    unconditionally with no terminal check either.
    """

    next_actions: torch.Tensor


@dataclass
class ChunkedReplayBufferSample(NStepReplayBufferSample):
    """Action-chunked (Q-chunking) replay sample.

    ``actions`` here is the full, uncollapsed ``(B, horizon_length,
    action_dim)`` window (not the ``(B, action_dim)`` shape the base class
    docstring implies) -- chunked algorithms flatten it to
    ``(B, horizon_length * action_dim)`` at the loss call site. ``rewards``/
    ``discounts``/``dones`` are the collapsed H-step discounted return, same
    convention as ``NStepReplayBufferSample``. ``valid`` is a ``(B,
    horizon_length)`` per-position mask (1 up to and including the position
    of a true in-window terminal, 0 after) for chunked BC-style losses that
    need per-position masking within a kept window.
    """

    valid: torch.Tensor
