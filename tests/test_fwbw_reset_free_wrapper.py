"""Tests for FWBWResetFreeWrapper."""
from __future__ import annotations

import gymnasium as gym
import torch

from rl_garden.envs.wrappers.fwbw_reset_free import FWBWResetFreeWrapper


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self, terminated_sequence):
        self._terminated_sequence = list(terminated_sequence)

    def reset(self, **kwargs):
        return torch.zeros(1, 4), {}

    def step(self, action):
        del action
        terminated = self._terminated_sequence.pop(0)
        return torch.ones(1, 4), torch.tensor([0.0]), torch.tensor([terminated]), torch.tensor([False]), {}


def test_direction_defaults_to_forward_and_is_reported_on_reset():
    wrapped = FWBWResetFreeWrapper(_FakeEnv([]))
    _, info = wrapped.reset()
    assert info["fwbw_direction"] == "forward"
    assert wrapped.direction == "forward"


def test_direction_flips_on_termination_and_holds_otherwise():
    wrapped = FWBWResetFreeWrapper(_FakeEnv([False, True, False]))
    wrapped.reset()

    _, _, _, _, info = wrapped.step(None)
    assert info["fwbw_direction"] == "forward"  # no termination yet

    _, _, _, _, info = wrapped.step(None)
    assert info["fwbw_direction"] == "backward"  # flipped on success

    _, _, _, _, info = wrapped.step(None)
    assert info["fwbw_direction"] == "backward"  # holds until next success


def test_initial_direction_can_be_backward():
    wrapped = FWBWResetFreeWrapper(_FakeEnv([]), initial_direction="backward")
    assert wrapped.direction == "backward"
