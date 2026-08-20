"""Replay buffer with next-step actions, for ReBRAC's critic-side BC penalty.

See ``ReBRACReplayBufferSample``'s docstring (``rl_garden/common/types.py``)
for the exact semantics and the two documented, CORL-matching
simplifications (terminal-boundary spillover, buffer wraparound).
"""
from __future__ import annotations

import torch

from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.types import ReBRACReplayBufferSample


class ReBRACReplayBuffer(TensorReplayBuffer):
    def _index_batch(
        self, batch_inds: torch.Tensor, env_inds: torch.Tensor
    ) -> ReBRACReplayBufferSample:
        next_inds = (batch_inds + 1) % self.per_env_buffer_size
        next_actions = self.actions[next_inds, env_inds].to(self.sample_device)
        return ReBRACReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
            next_actions=next_actions,
        )
