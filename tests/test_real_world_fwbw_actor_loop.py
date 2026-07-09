"""Tests for FWBWActorLoop against a fake FWBW-wrapped env."""
from __future__ import annotations

import pytest
import torch

from rl_garden.real_world.actor_loop import FWBWActorLoop


class _FakeFWBWEnv:
    """Simulates FWBWResetFreeWrapper's info contract: direction flips after
    a configured number of steps within each direction."""

    num_envs = 1

    def __init__(self, flip_after: int = 2):
        self.flip_after = flip_after
        self._direction = "forward"
        self._steps_in_direction = 0
        self.reset_calls = 0

    def reset(self, **kwargs):
        self.reset_calls += 1
        self._steps_in_direction = 0
        return torch.zeros(1, 4), {"fwbw_direction": self._direction}

    def step(self, action):
        del action
        self._steps_in_direction += 1
        if self._steps_in_direction >= self.flip_after:
            self._direction = "backward" if self._direction == "forward" else "forward"
            self._steps_in_direction = 0
        return (
            torch.ones(1, 4),
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"fwbw_direction": self._direction},
        )


class _FakePolicy:
    def __init__(self, tag: float):
        self.tag = tag
        self.loaded = []

    def eval(self):
        pass

    def load_state_dict(self, sd):
        self.loaded.append(sd)

    def predict(self, obs, deterministic=False):
        del obs, deterministic
        return torch.full((1, 2), self.tag)


class _FakeSyncClient:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.pushed = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def push_transition(self, transition):
        self.pushed.append(transition)

    def latest_policy_params(self):
        return None


def test_fwbw_actor_loop_routes_transitions_to_matching_direction():
    env = _FakeFWBWEnv(flip_after=2)
    policies = {"forward": _FakePolicy(1.0), "backward": _FakePolicy(2.0)}
    sync_clients = {"forward": _FakeSyncClient(), "backward": _FakeSyncClient()}
    loop = FWBWActorLoop(env, policies, sync_clients, control_hz=1_000_000.0)

    loop.run(total_steps=4)

    # direction: step1=forward(count1), step2=forward(count2->flip to backward),
    # step3=backward(count1), step4=backward(count2->flip to forward)
    assert len(sync_clients["forward"].pushed) == 2
    assert len(sync_clients["backward"].pushed) == 2
    for c in sync_clients.values():
        assert c.started and c.stopped


def test_fwbw_actor_loop_uses_the_active_direction_policy_for_action():
    env = _FakeFWBWEnv(flip_after=100)  # never flips within this test
    policies = {"forward": _FakePolicy(1.0), "backward": _FakePolicy(2.0)}
    sync_clients = {"forward": _FakeSyncClient(), "backward": _FakeSyncClient()}
    loop = FWBWActorLoop(env, policies, sync_clients, control_hz=1_000_000.0)

    loop.run(total_steps=1)

    pushed = sync_clients["forward"].pushed
    assert len(pushed) == 1
    torch.testing.assert_close(pushed[0]["action"], torch.full((1, 2), 1.0))


def test_fwbw_actor_loop_requires_forward_and_backward_keys():
    env = _FakeFWBWEnv()
    with pytest.raises(ValueError, match="forward"):
        FWBWActorLoop(env, {"forward": _FakePolicy(1.0)}, {"forward": _FakeSyncClient()})
