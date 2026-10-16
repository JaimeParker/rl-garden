"""``SequenceBatch``: the time-major window batch every ``WorldModel.model_loss``
reads (attribute access, not dict keys).

Required fields:

- ``obs``: ``(T + 1, B, ...)`` observations (a tensor or a ``dict`` of tensors).
- ``action``: ``(T, B, A)``.
- ``reward``: ``(T, B, 1)``. Buffers that store rewards as ``(T, B)``
  (``SequenceReplayBuffer``'s strict and uniform modes) are accepted too.
  ``RSSM.model_loss`` adds the trailing dim itself.

Optional fields (``None`` when a model or caller does not need them):

- ``is_first``: ``(T, B)`` bool, recurrent reset flags (RSSM).
- ``carry``: ``{key: (B, ...)}``, recurrent warm start (RSSM).
- ``terminated``: ``(T, B)`` or ``(T, B, 1)`` termination flags.
- ``task``: ``(B,)`` task ids (``MultitaskWorldModel``).
- ``next_z``: ``(T, B, latent)`` no-grad target latents for the
  latent-consistency models. When absent, the model encodes ``obs[1:]`` itself
  under ``torch.no_grad()``.
- ``reward_valid``: ``(B,)`` bool. Marks the columns that carry a reward
  label. ``None`` means every column is valid. Otherwise the reward loss
  averages over the valid columns only.
- ``indices``: buffer indices for carry write-back (``SequenceReplayBuffer``
  uniform mode).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from rl_garden.common.types import Obs


@dataclass
class SequenceBatch:
    obs: Obs
    action: torch.Tensor
    reward: torch.Tensor
    is_first: Optional[torch.Tensor] = None
    carry: Optional[dict[str, torch.Tensor]] = None
    terminated: Optional[torch.Tensor] = None
    task: Optional[torch.Tensor] = None
    next_z: Optional[torch.Tensor] = None
    reward_valid: Optional[torch.Tensor] = None
    indices: Optional[Any] = None
