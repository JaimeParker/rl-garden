from __future__ import annotations

import torch

from rl_garden.networks.discrete_critic import DiscreteCritic


def test_forward_shape():
    critic = DiscreteCritic(features_dim=4, hidden_dims=[8], n_actions=3)
    q = critic(torch.randn(5, 4))
    assert q.shape == (5, 3)


def test_default_n_actions_is_three():
    critic = DiscreteCritic(features_dim=4, hidden_dims=[8])
    assert critic.n_actions == 3
    q = critic(torch.randn(2, 4))
    assert q.shape == (2, 3)
