"""Receding-horizon rollout wrapper for chunked (diffusion/flow) policies.

``DiffusionPolicy``/``VisionDiffusionPolicy``.predict() returns a full
``(B, horizon_steps, action_dim)`` chunk and leaves execution/slicing to the
caller (see those classes' own docstrings). This wraps one such policy to
expose the plain single-action ``BasePolicy.predict()`` contract instead:
sample a full chunk, execute the first ``n_action_steps`` actions from it one
at a time, then resample -- the same receding-horizon control pattern
``real-stanford/diffusion_policy``'s own ``predict_action`` uses (``start =
n_obs_steps - 1; end = start + n_action_steps``).
"""
from __future__ import annotations

from typing import Optional, Protocol

import torch

from rl_garden.common.types import Obs
from rl_garden.policies.base import BasePolicy


class _ChunkedPolicy(Protocol):
    horizon_steps: int
    cond_steps: int

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor: ...


class RecedingHorizonPolicy(BasePolicy):
    def __init__(self, policy: _ChunkedPolicy, *, n_action_steps: int) -> None:
        super().__init__()
        # Execution window is chunk[start : start + n_action_steps], where
        # start = cond_steps - 1 (matches real-stanford/diffusion_policy's
        # predict_action) -- so n_action_steps is bounded by the room left
        # in the chunk after that offset, not by horizon_steps alone.
        start = policy.cond_steps - 1
        max_n_action_steps = policy.horizon_steps - start
        if not (1 <= n_action_steps <= max_n_action_steps):
            raise ValueError(
                f"n_action_steps must be in [1, {max_n_action_steps}] "
                f"(horizon_steps={policy.horizon_steps}, cond_steps={policy.cond_steps}), "
                f"got {n_action_steps}."
            )
        self.policy = policy
        self.n_action_steps = n_action_steps
        self._start = start
        self._chunk: Optional[torch.Tensor] = None
        self._cursor = 0

    def reset_chunk(self) -> None:
        """Call on episode reset -- forces a fresh chunk sample on the next
        ``predict()`` call instead of continuing to drain a stale one."""
        self._chunk = None
        self._cursor = 0

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        if self._chunk is None or self._cursor >= self.n_action_steps:
            self._chunk = self.policy.predict(obs, deterministic=deterministic)
            self._cursor = 0
        action = self._chunk[:, self._start + self._cursor]
        self._cursor += 1
        return action
