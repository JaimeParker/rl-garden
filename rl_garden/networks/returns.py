"""``lambda_return``: the DreamerV3 TD(lambda) recurrence used to build value
targets from an imagined (or replayed) rollout.

Ported from the recurrence in ``3rd_party/r2dreamer/dreamer.py``'s
``_lambda_return`` (see that method's docstring: "lamb=1 means discounted
Monte Carlo return, lamb=0 means fixed 1-step return"), adapted from
r2dreamer's batch-major ``(B, T)`` two-signal (``is_last``/``is_terminal``)
form to the single-``continues`` time-major ``(T, B, ...)`` form used
throughout this repo's replay/rollout tensors (``rl_garden`` buffers are
``(T, N, ...)``, see ``AGENTS.md``). ``continues`` folds "is this transition
non-terminal" into one per-step survival probability (the caller is
responsible for producing it, e.g. ``1 - Bernoulli(cont_head).probs``, or
``(~terminated).float()`` for real replayed data) -- exactly one signal is
simpler than r2dreamer's two-signal split because rl-garden's replay
windows/imagined rollouts never cross an episode boundary internally (no
separate "is_last" needed), matching the classic Dreamer v1/v2
``tools.lambda_return`` recurrence.
"""
from __future__ import annotations

import torch


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    lam: float,
    gamma: float,
) -> torch.Tensor:
    """Computes the TD(``lam``) return target at every step of a rollout.

    All of ``rewards``, ``values``, ``continues`` are time-major ``(T, B,
    ...)`` (T aligned steps: ``rewards[t]``/``continues[t]`` describe the
    transition leaving step ``t``, ``values[t]`` is the value estimate at
    step ``t``). ``bootstrap`` is ``(B, ...)``, the value estimate for the
    (unobserved) state after step ``T-1``.

    Recurrence (right-to-left):
        ``pcont[t] = gamma * continues[t]``
        ``inputs[t] = rewards[t] + pcont[t] * next_values[t] * (1 - lam)``
        where ``next_values = cat([values[1:], bootstrap[None]], 0)``
        ``returns[T-1] = inputs[T-1] + pcont[T-1] * lam * bootstrap``
        ``returns[t]   = inputs[t]   + pcont[t]   * lam * returns[t + 1]``

    ``lam=0`` degenerates to the fixed 1-step bootstrapped return
    (``rewards[t] + pcont[t] * next_values[t]``); ``lam=1`` degenerates to
    the discounted Monte Carlo return (``sum_k gamma^k * rewards[t+k]`` plus
    a discounted bootstrap tail).

    Returns a ``(T, B, ...)`` tensor of return targets, one per input step.
    """
    if not (rewards.shape == values.shape == continues.shape):
        raise ValueError(
            "rewards/values/continues must share one (T, B, ...) shape, got "
            f"{tuple(rewards.shape)}/{tuple(values.shape)}/{tuple(continues.shape)}."
        )
    horizon = rewards.shape[0]
    pcont = gamma * continues
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    inputs = rewards + pcont * next_values * (1 - lam)

    returns = [bootstrap] * horizon  # placeholders, overwritten below
    last = bootstrap
    for t in reversed(range(horizon)):
        last = inputs[t] + pcont[t] * lam * last
        returns[t] = last
    return torch.stack(returns, dim=0)
