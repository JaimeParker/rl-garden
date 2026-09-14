"""Unit tests for ``rl_garden.networks.returns.lambda_return`` (DreamerV3
form, r2dreamer ``dreamer.py:553-566``) against hand-computed 4-step (time-
major ``(T=4, B=1)``) examples. All cases share ``reward = [_, 1, 2, 3]``
(index 0 unused -- shifted away by the recurrence's own ``[1:]`` slicing)
and ``boot = value = [_, 10, 20, 30]``, so only ``last``/``term``/``lam``
vary across cases; each expected value is derived by hand in the comment
above it.
"""
from __future__ import annotations

import pytest
import torch

from rl_garden.networks.returns import lambda_return


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)  # (T, 1)


def test_lambda_zero_is_fixed_one_step_bootstrapped_return():
    # lam=0 => cont = (1-last[1:])*0 == 0 everywhere, so interm collapses to
    # reward[1:] + live*boot[1:] and the recurrence's own cont term vanishes:
    # returns[t] = reward[t+1] + disc*boot[t+1] (t=0,1,2; disc=1 here).
    # returns[0] = reward[1] + boot[1] = 1 + 10 = 11
    # returns[1] = reward[2] + boot[2] = 2 + 20 = 22
    # returns[2] = reward[3] + boot[3] = 3 + 30 = 33
    reward = _t(0.0, 1.0, 2.0, 3.0)
    value = boot = _t(0.0, 10.0, 20.0, 30.0)
    last = term = torch.zeros(4, 1)

    returns = lambda_return(reward, value, boot, last=last, term=term, disc=1.0, lam=0.0)

    torch.testing.assert_close(returns, _t(11.0, 22.0, 33.0))


def test_lambda_one_is_undiscounted_monte_carlo_return():
    # lam=1, disc=1, no last/term: returns[t] = sum(reward[t+1:]) + boot[3].
    # returns[0] = 1+2+3+30 = 36
    # returns[1] =   2+3+30 = 35
    # returns[2] =     3+30 = 33
    reward = _t(0.0, 1.0, 2.0, 3.0)
    value = boot = _t(0.0, 10.0, 20.0, 30.0)
    last = term = torch.zeros(4, 1)

    returns = lambda_return(reward, value, boot, last=last, term=term, disc=1.0, lam=1.0)

    torch.testing.assert_close(returns, _t(36.0, 35.0, 33.0))


def test_last_set_mid_sequence_breaks_lambda_continuity_not_the_bootstrap():
    # last=[_, 0, 1, 0] (last[2]=1): cont[1] = (1-1)*lam = 0, so the
    # recursion stops bootstrapping THROUGH row 1's return via lambda -- but
    # live is untouched (no term set), so row 1's own one-step bootstrap
    # (its own `boot[2]`) still applies:
    # interm[0] = reward[1] + (1-cont[0])*live[0]*boot[1] = 1 + 0*10 = 1        (cont[0]=1)
    # interm[1] = reward[2] + (1-cont[1])*live[1]*boot[2] = 2 + 1*20 = 22       (cont[1]=0)
    # interm[2] = reward[3] + (1-cont[2])*live[2]*boot[3] = 3 + 0*30 = 3        (cont[2]=1)
    # returns[2] = interm[2] + live[2]*cont[2]*boot[3]       = 3 + 30      = 33
    # returns[1] = interm[1] + live[1]*cont[1]*returns[2]    = 22 + 0      = 22
    # returns[0] = interm[0] + live[0]*cont[0]*returns[1]    = 1 + 22      = 23
    reward = _t(0.0, 1.0, 2.0, 3.0)
    value = boot = _t(0.0, 10.0, 20.0, 30.0)
    last = _t(0.0, 0.0, 1.0, 0.0)
    term = torch.zeros(4, 1)

    returns = lambda_return(reward, value, boot, last=last, term=term, disc=1.0, lam=1.0)

    torch.testing.assert_close(returns, _t(23.0, 22.0, 33.0))


def test_term_set_mid_sequence_zeroes_the_value_bootstrap():
    # term=[_, 0, 1, 0] (term[2]=1): live[1] = (1-1)*disc = 0, so BOTH the
    # boot term feeding interm[1] and the recursive cont*live product at
    # index 1 are zeroed -- row 1's return is exactly its own reward, no
    # bootstrap in either direction (a true terminal state's value is its
    # reward alone):
    # interm[0] = reward[1] + (1-cont[0])*live[0]*boot[1] = 1 + 0*10 = 1   (cont[0]=1, live[0]=1)
    # interm[1] = reward[2] + (1-cont[1])*live[1]*boot[2] = 2 + 0*20 = 2   (cont[1]=1, live[1]=0)
    # interm[2] = reward[3] + (1-cont[2])*live[2]*boot[3] = 3 + 0*30 = 3   (cont[2]=1, live[2]=1)
    # returns[2] = interm[2] + live[2]*cont[2]*boot[3]    = 3 + 30       = 33
    # returns[1] = interm[1] + live[1]*cont[1]*returns[2] = 2 + 0        = 2
    # returns[0] = interm[0] + live[0]*cont[0]*returns[1] = 1 + 2        = 3
    reward = _t(0.0, 1.0, 2.0, 3.0)
    value = boot = _t(0.0, 10.0, 20.0, 30.0)
    last = torch.zeros(4, 1)
    term = _t(0.0, 0.0, 1.0, 0.0)

    returns = lambda_return(reward, value, boot, last=last, term=term, disc=1.0, lam=1.0)

    torch.testing.assert_close(returns, _t(3.0, 2.0, 33.0))


def test_value_argument_is_shape_checked_but_not_used_in_the_recurrence():
    # `value` mirrors r2dreamer's own unused-but-shape-asserted argument
    # (dreamer.py:553) -- passing a wrong-shaped value must raise, but any
    # correctly-shaped value (even one wildly different from `boot`) must
    # not change the result.
    reward = _t(0.0, 1.0, 2.0, 3.0)
    boot = _t(0.0, 10.0, 20.0, 30.0)
    last = term = torch.zeros(4, 1)

    with pytest.raises(ValueError):
        lambda_return(
            reward, torch.zeros(3, 1), boot, last=last, term=term, disc=1.0, lam=1.0
        )

    baseline = lambda_return(reward, boot, boot, last=last, term=term, disc=1.0, lam=1.0)
    unused_value = _t(-999.0, -999.0, -999.0, -999.0)
    other = lambda_return(reward, unused_value, boot, last=last, term=term, disc=1.0, lam=1.0)
    torch.testing.assert_close(baseline, other)


def test_lambda_return_output_is_one_shorter_than_the_input_and_finite():
    horizon, batch = 6, 4
    reward = torch.randn(horizon, batch)
    value = boot = torch.randn(horizon, batch)
    last = (torch.rand(horizon, batch) > 0.8).float()
    term = (torch.rand(horizon, batch) > 0.9).float()

    returns = lambda_return(reward, value, boot, last=last, term=term, disc=0.997, lam=0.95)

    assert returns.shape == (horizon - 1, batch)
    assert torch.isfinite(returns).all()
