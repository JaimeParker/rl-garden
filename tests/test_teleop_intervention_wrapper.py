"""Tests for TeleopInterventionWrapper against a fake teleop device."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.envs.wrappers.teleop_intervention import (
    TeleopInterventionVectorWrapper,
    TeleopInterventionWrapper,
)
from robot_infra.teleop.utils.telo_op_control_twist import TeleOpSample


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self, action_dim: int | None = None):
        self.reset_calls = 0
        self.step_actions = []
        if action_dim is not None:
            self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        self.reset_calls += 1
        return torch.zeros(1, 4), {}

    def step(self, action):
        self.step_actions.append(action)
        return torch.ones(1, 4), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _FakeVectorEnv(gym.vector.VectorEnv):
    def __init__(self, action_dim: int | None = None):
        self.reset_calls = 0
        self.step_actions = []
        observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        action_space = spaces.Box(
            -1.0,
            1.0,
            (7 if action_dim is None else action_dim,),
            dtype=np.float32,
        )
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.observation_space = batch_space(observation_space, 1)
        self.action_space = batch_space(action_space, 1)
        self.metadata = {}

    def reset(self, **kwargs):
        self.reset_calls += 1
        return torch.zeros(1, 4), {}

    def step(self, actions):
        self.step_actions.append(actions)
        return torch.ones(1, 4), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}

    def close(self, **kwargs):
        pass


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


def test_intervention_drops_gripper_when_record_gripper_false():
    env = _FakeEnv(action_dim=6)
    teleop = _FakeTeleop([_sample(intervened=True, action_value=5.0)])
    wrapped = TeleopInterventionWrapper(env, teleop=teleop, record_gripper=False)

    policy_action = torch.full((1, 6), 1.0)
    _, _, _, _, info = wrapped.step(policy_action)

    torch.testing.assert_close(env.step_actions[0], torch.full((1, 6), 5.0))
    torch.testing.assert_close(info["intervene_action"], torch.full((1, 6), 5.0))


def test_intervention_keeps_gripper_by_default_and_validates_shape():
    env = _FakeEnv(action_dim=6)
    teleop = _FakeTeleop([_sample(intervened=True, action_value=5.0)])

    with pytest.raises(ValueError, match="dimension mismatch"):
        TeleopInterventionWrapper(env, teleop=teleop)


def test_vector_env_no_intervention_passes_policy_action_through():
    env = _FakeVectorEnv()
    teleop = _FakeTeleop([_sample(intervened=False)])
    wrapped = TeleopInterventionVectorWrapper(env, teleop=teleop)

    policy_action = torch.full((1, 7), 1.0)
    _, _, _, _, info = wrapped.step(policy_action)

    torch.testing.assert_close(env.step_actions[0], policy_action)
    assert "intervene_action" not in info
    assert info["human_episode_end"] is False


def test_vector_env_intervention_overrides_action_and_flags_info():
    env = _FakeVectorEnv(action_dim=6)
    teleop = _FakeTeleop([_sample(intervened=True, action_value=5.0)])
    wrapped = TeleopInterventionVectorWrapper(env, teleop=teleop, record_gripper=False)

    policy_action = torch.full((1, 6), 1.0)
    _, _, _, _, info = wrapped.step(policy_action)

    torch.testing.assert_close(env.step_actions[0], torch.full((1, 6), 5.0))
    torch.testing.assert_close(info["intervene_action"], torch.full((1, 6), 5.0))


def test_vector_env_validates_gripper_shape_at_startup():
    env = _FakeVectorEnv(action_dim=6)
    teleop = _FakeTeleop([_sample(intervened=True, action_value=5.0)])

    with pytest.raises(ValueError, match="dimension mismatch"):
        TeleopInterventionVectorWrapper(env, teleop=teleop)


def test_reset_resets_teleop_device_too():
    env = _FakeEnv()
    teleop = _FakeTeleop([])
    wrapped = TeleopInterventionWrapper(env, teleop=teleop)
    wrapped.reset()
    assert teleop.reset_calls == 1
    assert env.reset_calls == 1
