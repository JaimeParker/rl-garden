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
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from rl_garden.world_models.base import State, WorldModel


@dataclass
class ImaginedTrajectory:
    """One imagined rollout, time-major throughout.

    ``states``: each tensor is ``(horizon + 1, B, ...)`` (index 0 is the
    real ``start_state`` passed to ``imagine()``, indices ``1..horizon`` are
    ``model.step()`` outputs).
    ``actions``/``rewards``/``continues``/``done_mask``: each ``(horizon, B,
    ...)`` -- step ``t`` is the action taken *from* ``states[t]``, and the
    reward/continue-probability/done-flag observed on the resulting
    transition into ``states[t + 1]``.
    ``done_mask[t]`` is 1.0 once the rollout has crossed a predicted
    termination at or before step ``t`` (cumulative), 0.0 while still alive
    -- multiply per-step losses by ``(1 - done_mask)`` to stop them past a
    predicted terminal state, matching ``rl_garden.planners.mppi``'s
    termination-clipping convention.
    """

    states: State
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    done_mask: torch.Tensor


def imagine(
    model: WorldModel,
    policy_fn: Callable[[State], torch.Tensor],
    start_state: State,
    horizon: int,
    *,
    grad: bool,
    termination_fn: Optional[Callable[[State], torch.Tensor]] = None,
) -> ImaginedTrajectory:
    """Rolls ``model`` forward ``horizon`` steps from ``start_state`` under
    actions sampled from ``policy_fn``, with no new observations (pure
    imagination -- every step is ``model.step()``, never ``model.observe()``).

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

    If ``model.continue_`` returns ``None`` (model has no learned
    continuation head) and no ``termination_fn`` is given, the rollout is
    treated as never terminating (``continues`` is all ones). Both
    ``model.continue_`` and ``termination_fn`` return a *continue
    probability* (``>= 0.5`` means "still alive"), not a terminated flag.
    """
    context = torch.no_grad() if not grad else contextlib.nullcontext()
    with context:
        state = start_state
        state_steps: list[State] = [state]
        actions: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        continues: list[torch.Tensor] = []
        done_masks: list[torch.Tensor] = []
        alive: Optional[torch.Tensor] = None

        for _ in range(horizon):
            action = policy_fn(state)
            reward = model.reward(state, action)
            next_state = model.step(state, action, sample=True)

            continue_prob = model.continue_(next_state)
            if continue_prob is None:
                continue_prob = (
                    termination_fn(next_state) if termination_fn is not None else torch.ones_like(reward)
                )

            alive = (
                (continue_prob >= 0.5).to(continue_prob.dtype)
                if alive is None
                else alive * (continue_prob >= 0.5).to(continue_prob.dtype)
            )

            actions.append(action)
            rewards.append(reward)
            continues.append(continue_prob)
            done_masks.append(1.0 - alive)
            state = next_state
            state_steps.append(state)

        states: State = {
            key: torch.stack([s[key] for s in state_steps], dim=0) for key in state_steps[0]
        }
        return ImaginedTrajectory(
            states=states,
            actions=torch.stack(actions, dim=0),
            rewards=torch.stack(rewards, dim=0),
            continues=torch.stack(continues, dim=0),
            done_mask=torch.stack(done_masks, dim=0),
        )
