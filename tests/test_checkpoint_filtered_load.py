"""Tests for the generic filtered-state-dict-loading primitive
(``rl_garden/common/checkpoint.py::load_filtered_state_dict``), added for
teacher-checkpoint loading in ``PolicyDistillation`` but usable by any
algorithm needing to load one submodule's weights from another checkpoint.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rl_garden.common.checkpoint import load_filtered_state_dict


class _Combo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)


def test_load_filtered_state_dict_loads_only_prefix_into_submodule():
    source = _Combo()
    with torch.no_grad():
        source.actor.weight.fill_(1.0)
        source.actor.bias.fill_(2.0)

    target = _Combo()
    original_critic_weight = target.critic.weight.clone()
    original_critic_bias = target.critic.bias.clone()

    load_filtered_state_dict(target.actor, source.state_dict(), prefix="actor")

    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.equal(target.actor.bias, source.actor.bias)
    # Untouched: prefix="actor" never reaches the critic.
    assert torch.equal(target.critic.weight, original_critic_weight)
    assert torch.equal(target.critic.bias, original_critic_bias)


def test_load_filtered_state_dict_empty_prefix_loads_whole_dict():
    source = _Combo()
    with torch.no_grad():
        source.actor.weight.fill_(5.0)
        source.critic.weight.fill_(6.0)

    target = _Combo()
    load_filtered_state_dict(target, source.state_dict(), prefix="")

    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.equal(target.critic.weight, source.critic.weight)


def test_load_filtered_state_dict_raises_on_unknown_prefix():
    source = _Combo()
    target = _Combo()
    with pytest.raises(ValueError, match="No state_dict keys found"):
        load_filtered_state_dict(target.actor, source.state_dict(), prefix="nonexistent")
