"""Rollout buffer variant that yields sequence-preserving minibatches for BPTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Union

import torch

from rl_garden.buffers.rollout_buffer import RolloutBuffer
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs

# Local, intentionally-duplicated type alias mirroring
# rl_garden.networks.recurrent.RecurrentState: rl_garden/buffers/ has zero
# dependency on rl_garden/networks/ today, and this file preserves that
# layering rather than importing the networks-layer type for one alias.
RecurrentState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


def _index_recurrent_state(state: RecurrentState, indices: torch.Tensor) -> RecurrentState:
    if isinstance(state, tuple):
        return tuple(h[:, indices] for h in state)
    return state[:, indices]


@dataclass
class RecurrentRolloutBufferSample:
    obs: Obs                       # (T, mb_envs, *obs_shape)
    actions: torch.Tensor          # (T, mb_envs, act_dim)
    old_values: torch.Tensor       # (T, mb_envs)
    old_log_prob: torch.Tensor     # (T, mb_envs)
    advantages: torch.Tensor       # (T, mb_envs)
    returns: torch.Tensor          # (T, mb_envs)
    episode_starts: torch.Tensor   # (T, mb_envs) -- 1.0 = reset hidden state before this timestep
    initial_hidden: RecurrentState # sliced to mb_envs


class RecurrentRolloutBuffer(RolloutBuffer):
    def get_sequences(
        self,
        num_minibatches: int,
        initial_hidden: RecurrentState,
        *,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Iterator[RecurrentRolloutBufferSample]:
        """Env-axis-only minibatching: every minibatch yields the FULL
        ``(T, mb_envs, ...)`` window, never subsetting the time axis. This is what
        lets ``initial_hidden`` (the true hidden state at the start of collection)
        be reused directly for BPTT with zero per-timestep hidden-state storage --
        see ``rl_garden.networks.recurrent.RecurrentLatentEncoder``'s module
        docstring for why this is on-policy-only.

        Requires ``self.num_envs % num_minibatches == 0``.
        """
        if not self.full:
            raise RuntimeError("RolloutBuffer must be full before sampling.")
        if self.num_envs % num_minibatches != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_minibatches "
                f"({num_minibatches}) for env-axis minibatching."
            )

        # episode_starts[t+1] = dones[t] (dones[t] means "episode ended as a
        # result of stepping from obs[t]"). Row 0 is always 0 here -- the true
        # boundary reset from the END of the previous rollout window is folded
        # into `initial_hidden` by the caller (see PPO's rollout loop), since
        # this buffer has no visibility into a done from a prior window.
        episode_starts = torch.cat(
            [torch.zeros(1, self.num_envs, device=self.device), self.dones[:-1]], dim=0
        )

        mb_envs = self.num_envs // num_minibatches
        env_indices = (
            torch.randperm(self.num_envs, device=self.device, generator=generator)
            if shuffle
            else torch.arange(self.num_envs, device=self.device)
        )

        for start in range(0, self.num_envs, mb_envs):
            idx = env_indices[start : start + mb_envs]
            yield RecurrentRolloutBufferSample(
                obs=index_obs(self.obs, (slice(None), idx)),
                actions=self.actions[:, idx],
                old_values=self.values[:, idx],
                old_log_prob=self.log_probs[:, idx],
                advantages=self.advantages[:, idx],
                returns=self.returns[:, idx],
                episode_starts=episode_starts[:, idx],
                initial_hidden=_index_recurrent_state(initial_hidden, idx),
            )


class RecurrentDictRolloutBuffer(RecurrentRolloutBuffer):
    """Alias class for SB3-style naming when observations are Dict spaces."""
