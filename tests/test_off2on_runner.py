"""Tests for off2on/_runner.py's Minari wiring helpers."""
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.training.off2on._runner import (
    _require_continuous_action_space,
    _resolve_env_id,
)


def _args(**overrides):
    defaults = dict(
        dataset_source="maniskill_h5",
        env_id="PickCube-v1",
        offline_dataset_path=None,
        env_backend="maniskill",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_env_id_binds_to_minari_dataset_when_default(monkeypatch):
    args = _args(
        dataset_source="minari",
        env_id="PickCube-v1",
        offline_dataset_path="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/umaze-v1"


def test_resolve_env_id_respects_explicit_override():
    args = _args(
        dataset_source="minari",
        env_id="D4RL/antmaze/large-play-v1",
        offline_dataset_path="D4RL/antmaze/umaze-v1",
    )
    assert _resolve_env_id(args) == "D4RL/antmaze/large-play-v1"


def test_resolve_env_id_unaffected_for_maniskill_h5():
    args = _args(dataset_source="maniskill_h5", env_id="PickCube-v1")
    assert _resolve_env_id(args) == "PickCube-v1"


def test_require_continuous_action_space_allows_box():
    env = SimpleNamespace(
        single_action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    )
    _require_continuous_action_space(env, _args())


def test_require_continuous_action_space_rejects_discrete():
    env = SimpleNamespace(single_action_space=spaces.Discrete(4))
    with pytest.raises(ValueError, match="Discrete"):
        _require_continuous_action_space(
            env, _args(env_backend="minari", env_id="atari/pong/expert-v0")
        )
