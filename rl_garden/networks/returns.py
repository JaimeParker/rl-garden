"""``lambda_return``: the DreamerV3 TD(lambda) recurrence used to build value
targets from an imagined (or replayed) rollout.

Ported verbatim (same recurrence, same argument semantics) from r2dreamer's
``Dreamer._lambda_return`` (``dreamer.py:553-566``; identical in the official
JAX ``agent.py:482-490``), converted from that method's batch-major ``(B,
T)`` layout to this repo's time-major ``(T, B, ...)`` layout (``rl_garden``
replay/rollout tensors are ``(T, N, ...)``, see ``AGENTS.md``).

Superseded design note (2026-09-14, plan model-based-base Part 2 / DreamerV3
section A): this module used to hold a simpler single-``continues`` TD(lam)
recurrence (still correct for that design, but not what Dreamer's own
actor-critic loss uses). Dreamer needs ``last`` (episode/replay-window
boundary) and ``term`` (true MDP termination) as two independent signals --
``last`` only breaks λ-continuity (stops bootstrapping *past* a boundary
that isn't a real death, e.g. a truncated episode or a replay-chunk seam),
while ``term`` breaks the value bootstrap itself (a real terminal state's
value is exactly its reward, never a future estimate) -- so the two-signal
form replaces it outright rather than growing a second function.
"""
from __future__ import annotations

import torch


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    boot: torch.Tensor,
    *,
    last: torch.Tensor,
    term: torch.Tensor,
    disc: float,
    lam: float,
) -> torch.Tensor:
    """Computes the TD(``lam``) return target at every step of a ``T``-step
    rollout/replay window, time-major throughout: ``reward``/``value``/
    ``boot``/``last``/``term`` all share one ``(T, B, ...)`` shape.

    ``value`` is asserted shape-compatible but otherwise **unused in the
    recurrence** -- kept only to mirror r2dreamer's own signature
    (``dreamer.py:553``, whose body never references its ``value`` argument
    either; both call sites there pass the *same* tensor as ``value`` and
    ``boot`` for the imagination target, and a different pair for the
    replay-based target). Callers wanting "the return at step 0" as a
    bootstrap elsewhere (r2dreamer's replay-side ``boot = ret[:, 0]``) read
    the returned tensor's first row themselves.

    Recurrence (right-to-left, r2dreamer ``dreamer.py:553-566``):
        ``live[t] = (1 - term[t+1]) * disc``
        ``cont[t] = (1 - last[t+1]) * lam``
        ``interm[t] = reward[t+1] + (1 - cont[t]) * live[t] * boot[t+1]``
        ``out[T-1] = boot[T-1]`` (the last row's own bootstrap, unchanged)
        ``out[t] = interm[t] + live[t] * cont[t] * out[t+1]`` for ``t = T-2..0``
    where the ``[t+1]``-shifted reads mean every one of ``reward``/``term``/
    ``last``/``boot``'s **row 0 never contributes** to the output -- it
    exists only to be shifted away, matching r2dreamer's own ``[:, 1:]``
    slicing. Returns a ``(T - 1, B, ...)`` tensor (one fewer row than the
    inputs): row ``t`` is the return target for input row ``t``, bootstrapped
    off row ``t + 1`` onward.

    ``lam=1`` degenerates to the discounted Monte Carlo return; ``lam=0`` to
    the fixed 1-step bootstrapped return (``reward[t+1] + live[t] * boot[t+1]``,
    since ``cont=0`` collapses ``interm``'s ``(1 - cont)`` factor to 1 and the
    recurrence's own ``cont`` factor to 0).
    """
    if not (reward.shape == value.shape == boot.shape == last.shape == term.shape):
        raise ValueError(
            "reward/value/boot/last/term must share one (T, B, ...) shape, got "
            f"{tuple(reward.shape)}/{tuple(value.shape)}/{tuple(boot.shape)}/"
            f"{tuple(last.shape)}/{tuple(term.shape)}."
        )
    live = (1.0 - term[1:]) * disc
    cont = (1.0 - last[1:]) * lam
    interm = reward[1:] + (1.0 - cont) * live * boot[1:]

    out = [boot[-1]]
    for t in reversed(range(live.shape[0])):
        out.append(interm[t] + live[t] * cont[t] * out[-1])
    out.reverse()
    return torch.stack(out[:-1], dim=0)
