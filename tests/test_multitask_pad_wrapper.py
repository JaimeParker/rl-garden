"""``MultitaskPadWrapper``: obs zero-padding, action truncation, task info."""
from __future__ import annotations

import pytest
import torch

from rl_garden.envs.wrappers.multitask_pad import MultitaskPadWrapper
from test_pwm_smoke import FakeStateVectorEnv


def _wrap(env, **overrides):
    params = dict(obs_dim=6, action_dim=3, task_idx=2, task_obs_dim=4, task_action_dim=2)
    params.update(overrides)
    return MultitaskPadWrapper(env, **params)


def test_pads_observations_and_reports_task():
    wrapper = _wrap(FakeStateVectorEnv(obs_dim=4, action_dim=2))
    assert wrapper.single_observation_space["state"].shape == (6,)
    assert wrapper.single_action_space.shape == (3,)
    assert wrapper.action_space.shape == (2, 3)

    obs, info = wrapper.reset()
    assert obs["state"].shape == (2, 6)
    assert torch.all(obs["state"][:, :4] == 1) and torch.all(obs["state"][:, 4:] == 0)
    assert torch.equal(info["task"], torch.full((2,), 2))

    obs, _, _, _, info = wrapper.step(torch.randn(2, 3))
    assert obs["state"].shape == (2, 6)
    assert torch.equal(info["task"], torch.full((2,), 2))


def test_truncates_actions_to_task_dims():
    env = FakeStateVectorEnv(obs_dim=4, action_dim=2)
    wrapper = _wrap(env)
    actions = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    wrapper.step(actions)
    assert torch.equal(env.last_actions, actions[:, :2])


def test_pads_final_observation():
    env = FakeStateVectorEnv(obs_dim=4, action_dim=2)
    original_step = env.step

    def step_with_final(actions):
        obs, reward, term, trunc, info = original_step(actions)
        return obs, reward, term, trunc, {"final_observation": {"state": torch.ones(2, 4)}}

    env.step = step_with_final
    _, _, _, _, info = _wrap(env).step(torch.zeros(2, 3))
    assert info["final_observation"]["state"].shape == (2, 6)


def test_rejects_env_dims_that_differ_from_the_task():
    with pytest.raises(ValueError, match="do not match the dataset's task dims"):
        _wrap(FakeStateVectorEnv(obs_dim=5, action_dim=2))
    with pytest.raises(ValueError, match="smaller than"):
        _wrap(FakeStateVectorEnv(obs_dim=4, action_dim=2), obs_dim=3, task_obs_dim=4)
