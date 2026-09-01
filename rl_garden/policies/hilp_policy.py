"""Thin ``nn.Module`` container for HILP's five sub-networks (``phi``-value,
its target, skill-value, skill-critic ensemble + target, skill-actor).

Not a ``BasePolicy`` -- that ABC's contract (a single ``features_extractor``
+ a `predict(obs, deterministic)` that maps *obs alone* to an action) doesn't
fit HILP's shape: there is no unified "policy" here, only networks jointly
consumed by ``HILP``'s own loss methods, and any action prediction needs an
extra ``skill`` argument ``BasePolicy.predict`` has no slot for. This class
exists only so ``BaseAlgorithm``'s existing ``self.policy.state_dict()``/
``.train()``/``.eval()`` machinery keeps working unmodified -- assigning each
network as a plain attribute already registers it as an ``nn.Module``
submodule, which is all ``state_dict()`` traversal needs.
"""
from __future__ import annotations

import torch.nn as nn

from rl_garden.networks import (
    EnsembleQCritic,
    GoalConditionedPhiValue,
    UnsquashedGaussianActor,
    ValueNetwork,
)


class HILPPolicy(nn.Module):
    def __init__(
        self,
        value: GoalConditionedPhiValue,
        value_target: GoalConditionedPhiValue,
        skill_value: ValueNetwork,
        skill_critic: EnsembleQCritic,
        skill_critic_target: EnsembleQCritic,
        skill_actor: UnsquashedGaussianActor,
    ) -> None:
        super().__init__()
        self.value = value
        self.value_target = value_target
        self.skill_value = skill_value
        self.skill_critic = skill_critic
        self.skill_critic_target = skill_critic_target
        self.skill_actor = skill_actor
