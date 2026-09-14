"""Unit tests for ``rl_garden.networks.returns.lambda_return`` against
hand-computed 3-step examples (the two degenerate cases named in its
docstring: ``lam=0`` -> fixed 1-step bootstrapped return, ``lam=1``/``gamma=1``
-> discounted Monte Carlo return with discount 1, both easy to verify by
hand)."""
from __future__ import annotations

import torch

from rl_garden.networks.returns import lambda_return


def test_lambda_zero_is_fixed_one_step_bootstrapped_return():
    # lam=0: returns[t] == rewards[t] + gamma * continues[t] * next_values[t],
    # i.e. plain one-step TD bootstrap, no recursion into future returns.
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.ones(3, 1)
    bootstrap = torch.tensor([40.0])

    returns = lambda_return(rewards, values, continues, bootstrap, lam=0.0, gamma=1.0)

    expected = torch.tensor([[1.0 + 20.0], [2.0 + 30.0], [3.0 + 40.0]])
    torch.testing.assert_close(returns, expected)


def test_lambda_one_gamma_one_is_undiscounted_monte_carlo_return():
    # lam=1, gamma=1: returns[t] == sum(rewards[t:]) + bootstrap, i.e. the
    # plain discounted-by-nothing Monte Carlo return with a bootstrap tail.
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.ones(3, 1)
    bootstrap = torch.tensor([40.0])

    returns = lambda_return(rewards, values, continues, bootstrap, lam=1.0, gamma=1.0)

    expected = torch.tensor([[46.0], [45.0], [43.0]])
    torch.testing.assert_close(returns, expected)


def test_lambda_return_zeroes_out_the_tail_when_continue_is_zero():
    # continues[1] == 0 should sever the recursion: returns[0] only sees
    # reward[0] plus the (zeroed) discounted continuation, i.e. no bootstrap
    # leaks back past a terminal step.
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    continues = torch.tensor([[1.0], [0.0], [1.0]])
    bootstrap = torch.tensor([40.0])

    returns = lambda_return(rewards, values, continues, bootstrap, lam=1.0, gamma=1.0)

    # returns[1] = reward[1] + pcont[1]*next_values[1]*(1-lam) + pcont[1]*lam*returns[2]
    #            = 2 + 0 + 0 = 2 (continues[1] == 0 zeroes both terms)
    # returns[0] = reward[0] + pcont[0]*next_values[0]*0 + pcont[0]*lam*returns[1] = 1 + 2 = 3
    expected = torch.tensor([[3.0], [2.0], [43.0]])
    torch.testing.assert_close(returns, expected)


def test_lambda_return_shape_matches_inputs():
    horizon, batch = 5, 4
    rewards = torch.randn(horizon, batch)
    values = torch.randn(horizon, batch)
    continues = torch.rand(horizon, batch)
    bootstrap = torch.randn(batch)

    returns = lambda_return(rewards, values, continues, bootstrap, lam=0.95, gamma=0.99)

    assert returns.shape == (horizon, batch)
    assert torch.isfinite(returns).all()
