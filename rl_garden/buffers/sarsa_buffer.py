"""Next-action tracking for Cal-QL's SARSA/FQE reference-value network.

Extends ``MCReplayBufferMixin`` with the dataset's actual next-step action,
needed to fit a reference Q-network via SARSA/TD regression instead of
Monte-Carlo return-to-go -- Cal-QL's own fix (paper Appendix D) for
continuing tasks like D4RL locomotion, where MC return-to-go is
truncation-biased near every artificial ``timeouts`` cutoff (see
``MCReplayBufferMixin`` and ``rl_garden.buffers.d4rl_legacy_dataset``'s
locomotion-family ``RuntimeWarning``).

This mirrors ``ReBRACReplayBuffer``'s next-action index-shift
(``rl_garden.buffers.rebrac_replay_buffer``), but adds the one guard that
pattern lacks and does not need for its own use case: an explicit
``next_action_valid`` mask, ``False`` wherever the shift would spill into
the following episode. ReBRAC relies on downstream ``(1-dones)`` masking at
true terminals, which is safe there; it is not safe here; because
locomotion's ``dones`` (TD mask) is terminations-only and almost always 0,
a naive shift would silently read the next episode's first action at every
~1000-step ``timeouts`` boundary.
"""
from __future__ import annotations

from rl_garden.buffers.mc_buffer import MCReplayBufferMixin
from rl_garden.buffers.tensor_buffer import TensorReplayBuffer
from rl_garden.common.types import SarsaMCReplayBufferSample


class SarsaReferenceMixin:
    """Mixin adding ``next_actions``/``next_action_valid`` to an MC buffer.

    Usage:
        class SarsaMCTensorReplayBuffer(SarsaReferenceMixin, MCReplayBufferMixin, TensorReplayBuffer):
            pass
    """

    def _index_batch(self, batch_inds, env_inds) -> SarsaMCReplayBufferSample:
        base = super()._index_batch(batch_inds, env_inds)
        next_inds = (batch_inds + 1) % self.per_env_buffer_size
        next_actions = self.actions[next_inds, env_inds].to(self.sample_device)
        # Reuses MCReplayBufferMixin's own episode-boundary tracking -- the
        # same barrier _build_mc_table() already stops its recursion at.
        next_action_valid = (~self._episode_end[batch_inds, env_inds]).to(
            self.sample_device
        )
        return SarsaMCReplayBufferSample(
            obs=base.obs,
            next_obs=base.next_obs,
            actions=base.actions,
            rewards=base.rewards,
            dones=base.dones,
            mc_returns=base.mc_returns,
            next_actions=next_actions,
            next_action_valid=next_action_valid,
        )


class SarsaMCTensorReplayBuffer(SarsaReferenceMixin, MCReplayBufferMixin, TensorReplayBuffer):
    """``MCTensorReplayBuffer`` plus next-action tracking for Cal-QL's SARSA
    reference-value network. Flat (non-Dict) observations only."""

    pass
