"""``imagine()``: the one shared "roll a ``WorldModel`` forward under a
policy, with no new observations" primitive every imagination-based
model-based algorithm (DreamerV3; a future PWM) builds its actor-critic
targets from. TD-MPC2's own decision-time planner (``rl_garden.planners
.mppi``) does its own specialized rollout instead (it needs per-candidate
elite re-weighting mid-rollout, not a single fixed-policy trajectory) and
does not use this helper.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
import torch.nn as nn

from rl_garden.world_models.base import State, WorldModel

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def clone_and_freeze(module: _ModuleT) -> _ModuleT:
    """Deep-copies ``module``, detaches every parameter
    (``requires_grad_(False)``) and puts it in ``eval()`` mode.

    DreamerV3's actor-critic update needs a frozen SNAPSHOT of the world
    model + actor + critic + slow critic to roll imagination forward and
    compute reward/continue/value TARGETS from, while a separate, live
    forward pass through the currently-training modules produces the actual
    losses (r2dreamer ``dreamer.py``'s ``_imagine``/``clone_and_freeze``
    pattern, scratchpad ``dreamer-code-survey.md`` section 10). Official JAX
    needs no such snapshot at all -- its functional autodiff (``nj.grad``)
    only ever differentiates the explicit module list it's given, so a
    plain forward pass through the "same" online parameters is already
    gradient-free with respect to everything else; this helper is a
    PyTorch-only necessity of ``nn.Module``'s implicit, always-on autograd.

    r2dreamer's own ``clone_and_freeze`` deep-copies ONCE (at construction
    and after ``.to()``) and then keeps every frozen parameter's ``.data``
    ALIASED to the same storage as its online counterpart, so an in-place
    optimizer step on the online parameter is silently visible through the
    frozen copy too -- correct, but relies on every future parameter update
    staying in-place, which is not a general guarantee for an arbitrary
    ``torch.optim.Optimizer``. This port instead calls ``clone_and_freeze``
    fresh every gradient step (this repo's own choice, not a numerical
    difference from upstream: the snapshot is still exactly "whatever the
    online parameters are worth right now, no gradient attached'') -- a
    real deep copy every step costs more compute than the aliasing trick,
    but it is correct regardless of how the optimizer mutates parameters.
    """
    frozen = copy.deepcopy(module)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    frozen.eval()
    return frozen


@dataclass
class ImaginedTrajectory:
    """One imagined rollout, time-major throughout.

    ``states``: each tensor is ``(horizon + 1, B, ...)`` (index 0 is the
    real ``start_state`` passed to ``imagine()``, indices ``1..horizon`` are
    ``model.step()`` outputs).
    ``actions``: ``(horizon, B, ...)`` -- step ``t`` is the action taken
    *from* ``states[t]`` to reach ``states[t + 1]``.

    Deliberately minimal (fixer pass, 2026-09-14): this used to also carry
    per-step ``rewards``/``continues``/``done_mask``, computed by calling
    ``model.reward``/``model.continue_`` once per step during the rollout.
    No consumer ever read them -- ``DreamerV3._update_actor_critic`` (the
    only caller so far) computes reward/continue/value targets itself,
    batched ONE call each over the whole ``states`` tensor with its own
    frozen heads, after the rollout finishes (cheaper than ``horizon``
    separate per-step head calls, and this generic helper has no
    reward/continue head of its own to call anyway). A caller that still
    wants a per-step reward/continue trace computes it from the returned
    ``states``.
    """

    states: State
    actions: torch.Tensor


def imagine(
    model: WorldModel,
    policy_fn: Callable[[State], torch.Tensor],
    start_state: State,
    horizon: int,
    *,
    grad: bool,
) -> ImaginedTrajectory:
    """Rolls ``model`` forward ``horizon`` steps from ``start_state`` under
    actions sampled from ``policy_fn``, with no new observations (pure
    imagination -- every step is ``model.step()``, never ``model.observe()``).

    Returns only ``states``/``actions`` -- the minimum every consumer needs
    (see ``ImaginedTrajectory``'s docstring for why reward/continue/done-mask
    were dropped from here). A caller building reward/continue/value targets
    from the result computes them batched over the returned ``states`` with
    its own heads, exactly as ``DreamerV3._update_actor_critic`` does
    (``model.reward_head``/``model.cont_head``/the critic, each called once
    on ``model.features(traj.states)``, never once per step inside this
    rollout).

    ``start_state`` tensors are ``(B, ...)``. For Dreamer-style reuse, flatten
    a ``(T_data, B_data, ...)`` batch of posterior states from
    ``model.model_loss()`` into one ``(T_data * B_data, ...)`` batch before
    calling this -- Dreamer's own convention is that imagination starts from
    *every* posterior state of a training batch, flattened, not just each
    sequence's last step (scratchpad ``dreamer-code-survey.md`` section 10).
    ``model.model_loss()``'s returned posterior states are live (graph-
    attached, see ``WorldModel.model_loss``'s docstring) -- a caller building
    ``start_state`` for a ``grad=False`` imagination pass must ``.detach()``
    them itself (this helper's own ``torch.no_grad()`` only stops *new* graph
    nodes from being recorded during the rollout; it does not retroactively
    detach a ``start_state`` that already carries a live graph into it).

    ``grad=False`` wraps the whole rollout in ``torch.no_grad()`` (rollouts
    used only to build fixed value/policy targets). ``grad=True`` leaves
    autograd on -- **the caller is responsible for passing a frozen
    (stop-gradient) snapshot of ``model``/``policy_fn`` when gradients
    reaching their live parameters would be wrong**; this helper never clones
    or freezes anything itself. This mirrors Dreamer's own two-pass pattern
    (roll out under a frozen snapshot for the *targets*, then re-forward the
    resulting states through the *trainable* actor/critic for the losses --
    see ``dreamer-code-survey.md`` section 10's discussion of
    ``clone_and_freeze``, a PyTorch-only necessity with no JAX-side
    equivalent since functional autodiff there never touches parameters
    outside an explicit ``nj.grad`` call).
    """
    context = torch.no_grad() if not grad else contextlib.nullcontext()
    with context:
        state = start_state
        state_steps: list[State] = [state]
        actions: list[torch.Tensor] = []

        for _ in range(horizon):
            action = policy_fn(state)
            state = model.step(state, action, sample=True)
            actions.append(action)
            state_steps.append(state)

        states: State = {
            key: torch.stack([s[key] for s in state_steps], dim=0) for key in state_steps[0]
        }
        return ImaginedTrajectory(states=states, actions=torch.stack(actions, dim=0))
