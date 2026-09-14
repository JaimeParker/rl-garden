"""Tests for ``rl_garden.world_models.imagine.imagine()`` (split out of
``tests/test_world_models.py`` 2026-09-14 when the helper's contract was
simplified to ``states``/``actions`` only -- see that module's docstring).
No simulator/hardware -- fake CPU tensors only.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.world_models.base import State, WorldModel
from rl_garden.world_models.imagine import clone_and_freeze, imagine


class _FakeWorldModel(WorldModel):
    """Minimal concrete ``WorldModel``: state = {"h": Tensor}, a linear
    scalar-sum dynamics, used only to exercise ``imagine()`` without any
    real network."""

    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.latent_dim = dim
        self.linear = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.linear.weight)
        self.encoder = FlattenExtractor(spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder.extract(obs)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        return {"h": torch.zeros(batch_size, self.dim, device=device)}

    def observe(self, state, action, embed, is_first) -> State:
        del state, action, is_first
        return {"h": embed}

    def step(self, state: State, action: torch.Tensor, *, sample: bool = True) -> State:
        del sample
        return {"h": self.linear(state["h"]) + action}

    def reward(self, state: State, action: torch.Tensor) -> torch.Tensor:
        return state["h"].sum(-1, keepdim=True) + action.sum(-1, keepdim=True)

    def continue_(self, state: State):
        return torch.full((state["h"].shape[0], 1), 0.9)

    def features(self, state: State) -> torch.Tensor:
        return state["h"]

    def parameter_groups(self):
        return {"encoder": self.encoder.parameters(), "model": self.linear.parameters()}

    def model_loss(self, batch):
        raise NotImplementedError("not exercised by these tests")


def _policy_fn(state: State) -> torch.Tensor:
    return torch.ones_like(state["h"])


def test_imagine_produces_expected_shapes():
    model = _FakeWorldModel(dim=3)
    start = model.initial_state(batch_size=5, device=torch.device("cpu"))

    traj = imagine(model, _policy_fn, start, horizon=4, grad=False)

    assert traj.states["h"].shape == (5, 5, 3)  # horizon + 1, B, dim
    assert traj.actions.shape == (4, 5, 3)
    # Index 0 is the real start_state, unchanged; steps 1..horizon are model.step() outputs.
    torch.testing.assert_close(traj.states["h"][0], start["h"])


def test_imagine_grad_false_produces_no_grad_outputs():
    model = _FakeWorldModel(dim=2)
    model.linear.weight.requires_grad_(True)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))

    traj = imagine(model, _policy_fn, start, horizon=3, grad=False)

    assert not traj.states["h"].requires_grad
    assert not traj.actions.requires_grad


def test_imagine_grad_true_reaches_model_parameters():
    model = _FakeWorldModel(dim=2)
    model.linear.weight.requires_grad_(True)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))

    traj = imagine(model, _policy_fn, start, horizon=3, grad=True)
    assert traj.states["h"].requires_grad
    traj.states["h"].sum().backward()

    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()


def test_imagine_grad_true_with_frozen_snapshot_has_no_grad_into_original():
    """Mirrors ``DreamerV3._update_actor_critic``'s own pattern: a
    ``grad=False`` rollout under a ``clone_and_freeze``-d snapshot leaves the
    ORIGINAL (live) model's parameters completely untouched -- no ``.grad``
    at all, not even a zero one, since the frozen copy shares no storage
    with the original after ``clone_and_freeze``'s deep copy."""
    model = _FakeWorldModel(dim=2)
    model.linear.weight.requires_grad_(True)
    frozen = clone_and_freeze(model)
    start = model.initial_state(batch_size=2, device=torch.device("cpu"))

    traj = imagine(frozen, _policy_fn, start, horizon=3, grad=False)
    assert not traj.states["h"].requires_grad
    assert model.linear.weight.grad is None


def test_imagine_calls_policy_fn_horizon_times_with_the_current_state():
    model = _FakeWorldModel(dim=2)
    start = model.initial_state(batch_size=3, device=torch.device("cpu"))
    horizon = 5
    seen_states: list[State] = []

    def spy_policy_fn(state: State) -> torch.Tensor:
        seen_states.append(state)
        return torch.ones_like(state["h"])

    traj = imagine(model, spy_policy_fn, start, horizon=horizon, grad=False)

    assert len(seen_states) == horizon
    # policy_fn's t-th call must see states[t] (the state it acted FROM),
    # not states[t + 1] -- traj.states has horizon + 1 rows, the last of
    # which (the final model.step() output) is never passed to policy_fn.
    for t in range(horizon):
        torch.testing.assert_close(seen_states[t]["h"], traj.states["h"][t])
