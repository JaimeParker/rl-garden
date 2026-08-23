"""Tests for MultiStageBinaryRewardClassifierWrapper."""
from __future__ import annotations

import gymnasium as gym
import torch

from rl_garden.envs.wrappers.multi_stage_classifier import (
    MultiStageBinaryRewardClassifierWrapper,
)


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
            {},
        )


def test_each_stage_fires_exactly_once():
    # Stage 0 crosses threshold on step 1 only; stage 1 crosses on steps 1
    # and 2 (must not re-fire the second time).
    stage0_probs = iter([0.9, 0.9])
    stage1_probs = iter([0.9, 0.9])
    fns = [
        lambda obs: torch.tensor([next(stage0_probs)]),
        lambda obs: torch.tensor([next(stage1_probs)]),
    ]
    wrapped = MultiStageBinaryRewardClassifierWrapper(_FakeEnv(), fns, threshold=0.75)
    wrapped.reset()

    _, reward, terminated, _, info = wrapped.step(torch.zeros(1, 7))
    assert reward.item() == 2.0  # both stages newly fired
    assert bool(terminated[0])  # all stages received -> terminated
    assert bool(info["succeed"][0])

    _, reward2, _, _, _ = wrapped.step(torch.zeros(1, 7))
    assert reward2.item() == 0.0  # neither stage can re-fire


def test_terminated_stays_false_until_all_stages_received():
    # Use a stateful counter to control which stage fires on which step.
    call_count = {"n": 0}

    def stage0(obs):
        call_count["n"] += 1
        return torch.tensor([0.9 if call_count["n"] == 1 else 0.0])

    def stage1(obs):
        return torch.tensor([0.9 if call_count["n"] == 2 else 0.0])

    wrapped = MultiStageBinaryRewardClassifierWrapper(
        _FakeEnv(), [stage0, stage1], threshold=0.75
    )
    wrapped.reset()

    _, _, terminated1, _, info1 = wrapped.step(torch.zeros(1, 7))
    assert not bool(terminated1[0])
    assert not bool(info1["succeed"][0])

    _, _, terminated2, _, info2 = wrapped.step(torch.zeros(1, 7))
    assert bool(terminated2[0])
    assert bool(info2["succeed"][0])


def test_reset_clears_received_state():
    fired = {"n": 0}

    def classifier(obs):
        fired["n"] += 1
        return torch.tensor([0.9])

    wrapped = MultiStageBinaryRewardClassifierWrapper(_FakeEnv(), [classifier], threshold=0.75)
    wrapped.reset()
    wrapped.step(torch.zeros(1, 7))  # fires the only stage

    obs, info = wrapped.reset()
    assert not bool(info["succeed"][0])
    _, reward, terminated, _, _ = wrapped.step(torch.zeros(1, 7))
    assert reward.item() == 1.0  # fires again after reset
    assert bool(terminated[0])
