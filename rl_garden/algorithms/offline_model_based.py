"""``OfflineModelBasedAlgorithm``: the offline (dataset-driven) host for
model-based algorithms.

Same three-hook gradient step as the online ``ModelBasedAlgorithm``
(``ModelBasedStepMixin``, see ``rl_garden.algorithms._model_based_step``),
on ``OfflineRLAlgorithm`` instead of ``OffPolicyAlgorithm``: there is no
rollout, ``learn()`` drives ``train(gradient_steps)`` through
``run_offline_pretraining``, and ``_sample_train_batch()`` draws from the
offline buffer the subclass builds in ``_setup_model()``. Checkpointing uses
``BaseAlgorithm``'s hooks unchanged (``_optimizer_names``,
``_checkpoint_metadata``, ...).
"""
from __future__ import annotations

from rl_garden.algorithms._model_based_step import ModelBasedStepMixin
from rl_garden.algorithms.offline import OfflineRLAlgorithm


class OfflineModelBasedAlgorithm(ModelBasedStepMixin, OfflineRLAlgorithm):
    pass
