"""Tests for the ScriptedExpert protocol used by DAgger."""
from __future__ import annotations

import torch

from rl_garden.common.scripted_expert import ScriptedExpert


class _LinearExpert:
    """Toy scripted expert: a fixed linear controller, structurally
    satisfying ScriptedExpert without inheriting from it."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs @ self.weight


def _accepts_expert(expert: ScriptedExpert, obs: torch.Tensor) -> torch.Tensor:
    return expert(obs)


def test_plain_callable_satisfies_protocol_structurally():
    weight = torch.eye(3)
    expert = _LinearExpert(weight)
    obs = torch.randn(5, 3)
    action = _accepts_expert(expert, obs)
    assert torch.allclose(action, obs)


def test_lambda_also_satisfies_protocol():
    expert = lambda obs: torch.zeros(obs.shape[0], 2)  # noqa: E731
    obs = torch.randn(4, 6)
    action = _accepts_expert(expert, obs)
    assert action.shape == (4, 2)
