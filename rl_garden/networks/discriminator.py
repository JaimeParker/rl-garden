"""GAIL discriminator network: a state-action MLP producing a single logit,
matching ``imitation``'s own default ``BasicRewardNet(use_state=True,
use_action=True, use_next_state=False, use_done=False)`` configuration
(``3rd_party/imitation/src/imitation/rewards/reward_nets.py``). A high logit
means "looks like an expert transition."

``features_dim``, not an observation space: the discriminator is a
critic-role consumer of the actor/critic extractor contract (see
``rl_garden.policies.base.BasePolicy.extract_critic_features`` and
``rl_garden/algorithms/gail.py``'s ``GAIL._setup_model``/
``_discriminator_reward``/``_train_discriminator_step``) -- its caller
extracts critic-role features first, so this class never sees a raw
observation and needs no Box/Dict restriction of its own. Whenever the
critic (or shared) extractor's own ``encoder_config.normalize_obs`` is set,
those features are already running-mean/std normalized before reaching
here (``RunningObsNormalizer``, applied inside the extractor's own
``forward``/``extract``) -- this class has no normalization of its own.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.mlp import KernelInit, create_mlp


class GAILDiscriminator(nn.Module):
    """``forward(features, action) -> logit``."""

    def __init__(
        self,
        features_dim: int,
        action_space: spaces.Box,
        net_arch: Sequence[int] = (32, 32),
        kernel_init: Optional[KernelInit] = None,
    ) -> None:
        super().__init__()
        action_dim = int(torch.tensor(action_space.shape).prod().item())
        self.mlp = create_mlp(
            features_dim + action_dim,
            1,
            net_arch,
            kernel_init=kernel_init,
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([torch.flatten(features, 1), torch.flatten(action, 1)], dim=1)
        return self.mlp(inputs).squeeze(-1)
