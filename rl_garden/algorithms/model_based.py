"""``ModelBasedAlgorithm``: reusable online model-based RL base. Sits on ``OffPolicyAlgorithm`` (its rollout,
replay-buffer wiring, checkpointing, and eval loop are exactly the vectorized
GPU-native off-policy loop every other online algorithm in this package
uses) and adds the two things a model-based algorithm needs beyond a normal
off-policy algorithm:

1. **A ``WorldModel`` (``self.world_model``, via ``self.policy.world_model``
   -- the model is the policy's stateful actor extractor, see
   ``rl_garden.policies.tdmpc2_policy.TDMPC2Policy``) and a sequence replay
   buffer (``rl_garden.buffers.sequence_replay_buffer.SequenceReplayBuffer``,
   built by the subclass's ``_setup_model()``). DreamerV3 adds
   rollout-carried model ``State`` here, wired through ``observe()`` --
   TD-MPC2's own ``LatentConsistencyModel`` has no recurrent carry to reset
   between rollout steps, and its policy's ``predict()`` self-manages its
   own planner state instead (see ``TDMPC2Policy.predict()``), so this base
   does not carry any rollout state of its own yet.

2. **The three-hook gradient step** (``ModelBasedStepMixin``,
   ``rl_garden.algorithms._model_based_step``, shared with the offline host
   ``OfflineModelBasedAlgorithm``): ``train(gradient_steps)`` samples one
   window batch per step (``_sample_train_batch()``, default
   ``self.replay_buffer.sample(self.batch_size)``) and calls
   ``_update_model(batch) -> (metrics, posterior)``, then
   ``_update_actor_critic(batch, posterior) -> metrics``, then
   ``_update_targets()`` -- every subclass overrides exactly these three
   (never ``train()`` itself). ``posterior`` is whatever LIVE (graph-
   attached, see ``rl_garden.world_models.base.WorldModel.model_loss``'s
   docstring) ``State`` ``_update_model`` returned; ``_update_actor_critic``
   detaches it itself if its actor update must not also backprop into an
   already-stepped model (TD-MPC2's case, see ``TDMPC2._gradient_step``).

Optimizer ownership is entirely per-subclass (``_optimizer_names()``,
inherited from ``BaseAlgorithm``) -- this base does not construct any
optimizer itself, matching every non-model-based algorithm's convention.

``_on_learning_starts()`` (``OffPolicyAlgorithm``'s hook, default no-op) is
where a subclass needing a pretrain burst exactly at the ``learning_starts``
boundary (TD-MPC2's ``seed_steps`` semantics) overrides -- see that hook's
docstring on ``OffPolicyAlgorithm`` for why the burst has to live there
rather than in this class's ``train()``.
"""
from __future__ import annotations

from typing import Any

import torch

from rl_garden.algorithms._model_based_step import ModelBasedStepMixin
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm


class ModelBasedAlgorithm(ModelBasedStepMixin, OffPolicyAlgorithm):
    def _replay_buffer_step_kwargs(
        self, terminations: torch.Tensor, truncations: torch.Tensor
    ) -> dict[str, Any]:
        """``SequenceReplayBuffer.add()`` (both ``cross_episode`` modes)
        needs ``episode_end`` -- default for every model-based algorithm
        using it; a subclass with a differently-shaped buffer overrides."""
        return {"episode_end": terminations | truncations}
