"""RecurrentPPOPolicy with a two-head action mean and an auxiliary BCE head."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.buffers.rollout_buffer import ExtraSpecs
from rl_garden.policies import RecurrentPPOPolicy
from rlg_downstream_example.actor import SharedInputSplitActor

#: Per-step env info used as the auxiliary target, ``(N, 1)`` float.
AUX_TARGET_KEY = "aux_target"


class AuxSplitRecurrentPolicy(RecurrentPPOPolicy):
    """The action mean is two heads on the recurrent latent (components
    ``[0, action_split)`` through the ``pi`` MLP, the rest through one linear
    layer); ``aux_head`` maps the same latent to one logit."""

    #: The update's evaluate calls return ``{"latent": <the actor's features>}``.
    evaluation_outputs = True

    def __init__(self, *args: Any, action_split: int, aux_hidden: int, **kwargs: Any) -> None:
        # Read by _build_actor, which super().__init__() calls.
        self.action_split = action_split
        super().__init__(*args, **kwargs)
        # A policy submodule, so the optimizer, the gradient clip and the
        # checkpoint all cover it.
        width = self.recurrent_encoder.features_dim
        self.aux_head = nn.Sequential(nn.Linear(width, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, 1))

    def _build_actor(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Any,
        **actor_options: Any,
    ) -> nn.Module:
        # features_dim is the recurrent latent's width, which the actor reads.
        act_dim = int(np.prod(action_space.shape))
        return SharedInputSplitActor(
            [features_dim, features_dim],
            action_space,
            [(0, self.action_split), (self.action_split, act_dim)],
            [hidden_dims, []],
            **actor_options,
        )

    def rollout_extra_specs(self) -> ExtraSpecs:
        return {**super().rollout_extra_specs(), AUX_TARGET_KEY: ((1,), torch.float32)}
