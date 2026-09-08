"""Uni-O4's deployed policy: greedy self-confidence selection over an ensemble.

Faithful port of ``AdaptiveBehaviorProximalPolicyOptimization
.mixed_offline_evaluate``'s greedy mode
(``3rd_party/Uni-O4/abppo.py``): at each state, every ensemble member
proposes its own deterministic action and scores that action under its own
distribution (a self-confidence/peakiness measure); the member with the
highest self-log-prob "wins" and its action is used. This is Uni-O4's own
deployed/inference policy, not any single ensemble member evaluated alone.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rl_garden.common.types import Obs
from rl_garden.policies.base import BasePolicy
from rl_garden.policies.bc_policy import BCPolicy


class UniO4MixturePolicy(BasePolicy):
    """Wraps an ensemble of ``BCPolicy`` actors; ``predict`` picks the member
    most confident in its own (deterministic) action at each state.

    Also the natural checkpoint container for all ensemble members: since
    ``self.actors`` is an ``nn.ModuleList``, ``state_dict()`` serializes
    every member's weights under one ``actors.{i}.…`` namespace with no
    extra checkpoint-hook plumbing needed for the actors themselves.
    """

    def __init__(self, actors: Sequence[BCPolicy]) -> None:
        super().__init__()
        self.actors = nn.ModuleList(actors)

    def predict(self, obs: Obs, deterministic: bool = False) -> torch.Tensor:
        del deterministic  # member selection is always by deterministic self-confidence
        actions = []
        self_log_probs = []
        for actor in self.actors:
            features = actor.extract_features(obs)
            action = actor.actor.deterministic_action(features)
            log_prob = actor.actor.evaluate_action_log_prob(features, action).squeeze(-1)
            actions.append(action)
            self_log_probs.append(log_prob)
        actions_t = torch.stack(actions, dim=0)  # (N, batch, act_dim)
        scores_t = torch.stack(self_log_probs, dim=0)  # (N, batch)
        winner = scores_t.argmax(dim=0)  # (batch,)
        batch_idx = torch.arange(actions_t.shape[1], device=actions_t.device)
        return actions_t[winner, batch_idx]
