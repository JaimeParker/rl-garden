"""Tests for ActorLoop against a fake env/policy/sync client.

The fakes below only implement the surface ActorLoop actually calls
(``env.reset``/``env.step``/``env.num_envs``, ``policy.eval``/``predict``/
``load_state_dict``, ``sync_client.start``/``stop``/``push_transition``/
``latest_policy_params``) so a test failure here means ActorLoop itself
reached for something outside its documented contract.
"""
from __future__ import annotations

import pytest
import torch

from rl_garden.real_world.actor_loop import ActorLoop


class _FakeEnv:
    num_envs = 1

    def __init__(self, terminate_every: int | None = None):
        self.terminate_every = terminate_every
        self.reset_calls = 0
        self.step_calls = 0
        self._t = 0

    def reset(self, *, seed=None):
        del seed
        self.reset_calls += 1
        self._t = 0
        return torch.zeros(1, 4), {}

    def step(self, action):
        self.step_calls += 1
        self._t += 1
        next_obs = torch.full((1, 4), float(self.step_calls))
        reward = torch.tensor([1.0])
        terminated = torch.tensor(
            [self.terminate_every is not None and self._t % self.terminate_every == 0]
        )
        truncated = torch.tensor([False])
        return next_obs, reward, terminated, truncated, {}


class _FakePolicy:
    def __init__(self):
        self.eval_called = False
        self.loaded_state_dicts = []
        self.predict_calls = 0

    def eval(self):
        self.eval_called = True

    def load_state_dict(self, sd):
        self.loaded_state_dicts.append(sd)

    def predict(self, obs, deterministic=False):
        del obs, deterministic
        self.predict_calls += 1
        return torch.ones(1, 2)


class _FakeSyncClient:
    def __init__(self, params_to_serve=None):
        self.started = False
        self.stopped = False
        self.pushed = []
        self._params = params_to_serve

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def push_transition(self, transition):
        self.pushed.append(transition)

    def latest_policy_params(self):
        return self._params


def test_actor_loop_steps_env_and_pushes_transitions():
    env = _FakeEnv()
    policy = _FakePolicy()
    sync_client = _FakeSyncClient()
    loop = ActorLoop(env, policy, sync_client, control_hz=1_000_000.0)

    loop.run(total_steps=3)

    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert len(sync_client.pushed) == 3
    assert sync_client.started and sync_client.stopped
    assert policy.eval_called
    torch.testing.assert_close(sync_client.pushed[0]["obs"], torch.zeros(1, 4))
    torch.testing.assert_close(sync_client.pushed[0]["next_obs"], torch.full((1, 4), 1.0))


def test_actor_loop_resets_env_after_termination():
    env = _FakeEnv(terminate_every=2)
    policy = _FakePolicy()
    sync_client = _FakeSyncClient()
    loop = ActorLoop(env, policy, sync_client, control_hz=1_000_000.0)

    loop.run(total_steps=2)

    # step 2 terminates -> loop must reset before returning.
    assert env.reset_calls == 2


def test_actor_loop_refreshes_policy_params_each_step():
    env = _FakeEnv()
    policy = _FakePolicy()
    params = {"w": torch.tensor([1.0])}
    sync_client = _FakeSyncClient(params_to_serve=params)
    loop = ActorLoop(env, policy, sync_client, control_hz=1_000_000.0)

    loop.run(total_steps=2)

    assert len(policy.loaded_state_dicts) == 2
    assert policy.loaded_state_dicts[0] is params


def test_actor_loop_records_intervene_action_when_present():
    class _InterveningEnv(_FakeEnv):
        def step(self, action):
            next_obs, reward, terminated, truncated, _ = super().step(action)
            return next_obs, reward, terminated, truncated, {"intervene_action": torch.full((1, 2), 5.0)}

    env = _InterveningEnv()
    policy = _FakePolicy()
    sync_client = _FakeSyncClient()
    loop = ActorLoop(env, policy, sync_client, control_hz=1_000_000.0)

    loop.run(total_steps=1)

    torch.testing.assert_close(sync_client.pushed[0]["action"], torch.full((1, 2), 5.0))


def test_actor_loop_rejects_multi_env():
    class _MultiEnv(_FakeEnv):
        num_envs = 2

    with pytest.raises(ValueError, match="num_envs"):
        ActorLoop(_MultiEnv(), _FakePolicy(), _FakeSyncClient())
