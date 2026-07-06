"""Tests for the shared offline dataset dispatcher used by offline and off2on."""
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.training._dataset import infer_offline_dataset_specs, load_offline_dataset


def _args(**overrides):
    defaults = dict(
        dataset_source="maniskill_h5",
        offline_dataset_path="demos/pickcube.h5",
        offline_num_traj=None,
        action_low=-1.0,
        action_high=1.0,
        reward_scale=1.0,
        reward_bias=0.0,
        success_key=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _box_spaces():
    return (
        spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    )


def test_infer_specs_routes_to_h5(monkeypatch):
    obs_space, action_space = _box_spaces()
    called = {}

    def _fake_infer_specs_from_h5(path, *, action_low, action_high):
        called["path"] = path
        called["action_low"] = action_low
        called["action_high"] = action_high
        return obs_space, action_space

    monkeypatch.setattr(
        "rl_garden.training._dataset.infer_specs_from_h5", _fake_infer_specs_from_h5
    )

    result = infer_offline_dataset_specs(_args(dataset_source="maniskill_h5"))
    assert result == (obs_space, action_space)
    assert called == {"path": "demos/pickcube.h5", "action_low": -1.0, "action_high": 1.0}


def test_infer_specs_routes_to_minari(monkeypatch):
    obs_space, action_space = _box_spaces()
    called = {}

    def _fake_infer_specs_from_minari(dataset_id):
        called["dataset_id"] = dataset_id
        return obs_space, action_space

    monkeypatch.setattr(
        "rl_garden.training._dataset.infer_specs_from_minari",
        _fake_infer_specs_from_minari,
    )

    args = _args(dataset_source="minari", offline_dataset_path="D4RL/halfcheetah/medium-v0")
    result = infer_offline_dataset_specs(args)
    assert result == (obs_space, action_space)
    assert called == {"dataset_id": "D4RL/halfcheetah/medium-v0"}


def test_infer_specs_rejects_discrete_minari_action_space(monkeypatch):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    discrete_action_space = spaces.Discrete(4)

    monkeypatch.setattr(
        "rl_garden.training._dataset.infer_specs_from_minari",
        lambda dataset_id: (obs_space, discrete_action_space),
    )

    args = _args(dataset_source="minari", offline_dataset_path="atari/pong/expert-v0")
    with pytest.raises(ValueError, match="Discrete"):
        infer_offline_dataset_specs(args)


def test_infer_specs_unsupported_source_raises():
    with pytest.raises(ValueError, match="Unsupported offline dataset source"):
        infer_offline_dataset_specs(_args(dataset_source="bogus"))


def test_load_offline_dataset_routes_to_h5(monkeypatch):
    called = {}

    def _fake_load_maniskill_h5(replay_buffer, path, *, num_traj, reward_scale, reward_bias, success_key):
        called.update(
            replay_buffer=replay_buffer,
            path=path,
            num_traj=num_traj,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            success_key=success_key,
        )
        return 42

    monkeypatch.setattr(
        "rl_garden.training._dataset.load_maniskill_h5_to_replay_buffer",
        _fake_load_maniskill_h5,
    )

    buffer = object()
    loaded = load_offline_dataset(buffer, _args(dataset_source="maniskill_h5"))
    assert loaded == 42
    assert called["replay_buffer"] is buffer
    assert called["path"] == "demos/pickcube.h5"


def test_load_offline_dataset_routes_to_minari(monkeypatch):
    called = {}

    def _fake_load_minari(replay_buffer, dataset_id, *, num_episodes, reward_scale, reward_bias, success_key):
        called.update(
            replay_buffer=replay_buffer,
            dataset_id=dataset_id,
            num_episodes=num_episodes,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            success_key=success_key,
        )
        return 7

    monkeypatch.setattr(
        "rl_garden.training._dataset.load_minari_dataset_to_replay_buffer",
        _fake_load_minari,
    )

    buffer = object()
    args = _args(
        dataset_source="minari",
        offline_dataset_path="D4RL/halfcheetah/medium-v0",
        offline_num_traj=10,
    )
    loaded = load_offline_dataset(buffer, args)
    assert loaded == 7
    assert called["dataset_id"] == "D4RL/halfcheetah/medium-v0"
    assert called["num_episodes"] == 10


def test_load_offline_dataset_unsupported_source_raises():
    with pytest.raises(ValueError, match="Unsupported offline dataset source"):
        load_offline_dataset(object(), _args(dataset_source="bogus"))
