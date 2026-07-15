"""``BasePolicy`` wrapper so ``BaseAlgorithm.save()/load()/state_dict()`` work
unmodified for ``TDMPC2Multitask`` -- same rationale as
``rl_garden.algorithms.tdmpc2.policy.TDMPC2Policy``.

``predict()`` (action selection via the CEM planner) is out of scope for v1:
multitask training never touches a live env (see ``multitask/agent.py``'s
module docstring), so nothing calls it yet. It raises rather than silently
returning a wrong action, to be picked up explicitly when multitask
evaluation is added.
"""
from __future__ import annotations

import torch

from rl_garden.algorithms.tdmpc2.multitask.world_model import MultitaskWorldModel
from rl_garden.common.types import Obs
from rl_garden.policies.base import BasePolicy


class MultitaskTDMPC2Policy(BasePolicy):
    def __init__(self, world_model: MultitaskWorldModel) -> None:
        super().__init__()
        self.world_model = world_model

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError(
            "TDMPC2Multitask has no online rollout/eval path in this port "
            "(training is offline-only, see multitask/agent.py); predict() "
            "is unused until multitask evaluation is implemented."
        )
