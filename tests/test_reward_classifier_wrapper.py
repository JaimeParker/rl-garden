"""Tests for RewardClassifierWrapper."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper


class _FakeEnv(gym.Env):
    num_envs = 1

    def reset(self, **kwargs):
        return torch.zeros(1, 4), {}

    def step(self, action):
        del action
        return (
            torch.ones(1, 4),
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"raw": True},
        )


def test_success_yields_reward_one_and_terminates():
    classifier = lambda obs: torch.tensor([0.9])
    wrapped = RewardClassifierWrapper(_FakeEnv(), classifier, threshold=0.5)
    obs, reward, terminated, truncated, info = wrapped.step(torch.zeros(1, 7))
    assert reward.item() == 1.0
    assert bool(terminated[0])
    assert info["raw"] is True
    assert info["classifier_success_prob"].item() == pytest.approx(0.9)


def test_failure_yields_reward_zero_and_does_not_terminate():
    classifier = lambda obs: torch.tensor([0.1])
    wrapped = RewardClassifierWrapper(_FakeEnv(), classifier, threshold=0.5)
    _, reward, terminated, _, _ = wrapped.step(torch.zeros(1, 7))
    assert reward.item() == 0.0
    assert not bool(terminated[0])


def test_terminate_on_success_false_never_terminates():
    classifier = lambda obs: torch.tensor([0.99])
    wrapped = RewardClassifierWrapper(_FakeEnv(), classifier, threshold=0.5, terminate_on_success=False)
    _, reward, terminated, _, _ = wrapped.step(torch.zeros(1, 7))
    assert reward.item() == 1.0
    assert not bool(terminated[0])
