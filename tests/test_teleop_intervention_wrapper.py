"""Tests for TeleopInterventionWrapper against a fake teleop device."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from rl_garden.envs.wrappers.teleop_intervention import TeleopInterventionWrapper
from robot_infra.teleop.utils.telo_op_control_twist import TeleOpSample


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self.reset_calls = 0
        self.step_actions = []

    def reset(self, **kwargs):
        self.reset_calls += 1
        return torch.zeros(1, 4), {}

    def step(self, action):
        self.step_actions.append(action)
        return torch.ones(1, 4), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _FakeTeleop:
    def __init__(self, samples):
        self._samples = list(samples)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def poll(self):
        return self._samples.pop(0)


def _sample(intervened: bool, action_value: float = 9.0, episode_end: bool = False) -> TeleOpSample:
    return TeleOpSample(
        action=np.full(7, action_value, dtype=np.float32),
        twist=np.zeros(6, dtype=np.float32),
        gripper=1.0,
        bind_pressed=False,
        episode_end=episode_end,
        intervened=intervened,
    )


def test_no_intervention_passes_policy_action_through():
    env = _FakeEnv()
    teleop = _FakeTeleop([_sample(intervened=False)])
    wrapped = TeleopInterventionWrapper(env, teleop=teleop)

    policy_action = torch.full((1, 7), 1.0)
    obs, reward, terminated, truncated, info = wrapped.step(policy_action)

    torch.testing.assert_close(env.step_actions[0], policy_action)
    assert "intervene_action" not in info
    assert info["human_episode_end"] is False


def test_intervention_overrides_action_and_flags_info():
    env = _FakeEnv()
    teleop = _FakeTeleop([_sample(intervened=True, action_value=5.0, episode_end=True)])
    wrapped = TeleopInterventionWrapper(env, teleop=teleop)

    policy_action = torch.full((1, 7), 1.0)
    obs, reward, terminated, truncated, info = wrapped.step(policy_action)

    torch.testing.assert_close(env.step_actions[0], torch.full((1, 7), 5.0))
    torch.testing.assert_close(info["intervene_action"], torch.full((1, 7), 5.0))
    assert info["human_episode_end"] is True


def test_reset_resets_teleop_device_too():
    env = _FakeEnv()
    teleop = _FakeTeleop([])
    wrapped = TeleopInterventionWrapper(env, teleop=teleop)
    wrapped.reset()
    assert teleop.reset_calls == 1
    assert env.reset_calls == 1
