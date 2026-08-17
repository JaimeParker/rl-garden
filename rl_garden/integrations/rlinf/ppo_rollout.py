"""RLinf sync PPO rollout adapter.

Unlike Phase 2's ``sac_rollout.py`` (which bypasses RLinf's model registry
entirely, overriding ``init_worker``/``predict`` to build and call an
rl-garden policy directly), this adapter's model is registered with RLinf's
own ``get_model()`` dispatch (``ppo_model.register_rl_garden_ppo_model``) --
so ``MultiStepRolloutWorker.init_worker``
(``3rd_party/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py``, calls
``get_model(rollout_model_config)``) needs **no override at all**. This is a
direct consequence of going through RLinf's FSDP/model-registry machinery
(``ppo_actor.py``'s ``RLGardenPPOFSDPActor``) instead of bypassing it.

One override is still required: ``MultiStepRolloutWorker.predict`` branches
on ``SupportedModel(self.model_cfg.model_type) in [...]`` against a
hardcoded list of RLinf's own model-type members, which does not include
our new ``"rl_garden_ppo"`` key -- without an override, ``mode``/
``return_obs`` would not reach ``predict_action_batch`` correctly (a
silent-degradation risk during eval, not just a missing feature).

RLinf is optional: this module is importable without it (the class exists
with an ``object`` fallback base), but instantiating
``RLGardenPPORollout`` requires it.
"""
from __future__ import annotations

from typing import Any, Literal

import torch

from rl_garden.integrations.rlinf import require_rlinf

# Import-time registration side effect -- see the matching comment in
# ppo_actor.py. This worker's inherited init_worker() calls RLinf's own
# get_model(), which needs "rl_garden_ppo" registered in *this* Ray
# worker process's own copy of RLinf's _MODEL_REGISTRY, not just the
# driver's.
from rl_garden.integrations.rlinf import ppo_model  # noqa: F401

try:
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    _RLINF_AVAILABLE = True
except ImportError:
    MultiStepRolloutWorker = object  # type: ignore[assignment,misc]
    _RLINF_AVAILABLE = False


class RLGardenPPORollout(MultiStepRolloutWorker):
    def __init__(self, cfg: Any) -> None:
        require_rlinf()
        super().__init__(cfg)

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs, mode=mode, return_obs=True
            )
        result["expert_label_flag"] = False
        return actions, result
