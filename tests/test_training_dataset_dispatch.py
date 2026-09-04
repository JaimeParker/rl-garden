"""Tests for the shared offline dataset dispatcher used by offline and off2on.

Dispatch itself goes through ``rl_garden.buffers.dataset_backend_registry``
(no more if/elif in ``rl_garden.training._dataset``), so most of these tests
register a small fake backend directly in the registry (via
``monkeypatch.setitem`` on its module-global dict, auto-reverted after each
test) rather than monkeypatching per-format functions ``_dataset.py`` no
longer imports at all. A few tests still exercise a *real* registered
backend (``minari``/``d4rl_legacy``/``rlbench``) with just its underlying
per-format function monkeypatched away, to keep regression coverage for
each backend's own adapter behavior (Box-space rejection, ``num_traj`` ->
``num_episodes`` renaming, and the ``obs_mode``/``backend_config``
passthrough bugfix).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.buffers import dataset_backend_registry
from rl_garden.buffers.dataset_backend_registry import DatasetBackend, DatasetRequest
from rl_garden.training._dataset import infer_offline_dataset_specs, load_offline_dataset


def _args(**overrides):
    defaults = dict(
        dataset_backend="h5",
        offline_dataset="demos/pickcube.h5",
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


def _register_fake_backend(monkeypatch, name, *, infer_specs=None, load=None):
    class _FakeBackend(DatasetBackend):
        @classmethod
        def infer_specs(cls, req: DatasetRequest):
            return infer_specs(req)

        @classmethod
        def load(cls, buffer, req: DatasetRequest) -> int:
            return load(buffer, req)

    monkeypatch.setitem(dataset_backend_registry._REGISTRY, name, _FakeBackend)


def test_infer_specs_passes_the_full_request_through_to_the_registered_backend(monkeypatch):
    obs_space, action_space = _box_spaces()
    captured = {}

    def _infer_specs(req):
        captured["req"] = req
        return obs_space, action_space

    _register_fake_backend(monkeypatch, "_test_fake", infer_specs=_infer_specs)

    args = _args(
        dataset_backend="_test_fake",
        offline_dataset="demos/pickcube.h5",
        action_low=-2.0,
        action_high=2.0,
    )
    result = infer_offline_dataset_specs(args)

    assert result == (obs_space, action_space)
    req = captured["req"]
    assert req.path == "demos/pickcube.h5"
    assert req.action_low == -2.0
    assert req.action_high == 2.0


def test_load_offline_dataset_passes_the_full_request_through_to_the_registered_backend(
    monkeypatch,
):
    captured = {}

    def _load(buffer, req):
        captured["buffer"] = buffer
        captured["req"] = req
        return 42

    _register_fake_backend(monkeypatch, "_test_fake", load=_load)

    buffer = object()
    args = _args(
        dataset_backend="_test_fake",
        offline_dataset="demos/pickcube.h5",
        offline_num_traj=10,
        reward_scale=2.0,
        reward_bias=0.5,
        success_key="is_success",
    )
    loaded = load_offline_dataset(buffer, args)

    assert loaded == 42
    assert captured["buffer"] is buffer
    req = captured["req"]
    assert req.path == "demos/pickcube.h5"
    assert req.num_traj == 10
    assert req.reward_scale == 2.0
    assert req.reward_bias == 0.5
    assert req.success_key == "is_success"


def test_backend_config_looked_up_by_dataset_backend_name_not_env_backend(monkeypatch):
    """Regression test: backend_config must come from getattr(args,
    dataset_backend, None), independent of any --env_backend, since the two
    can legitimately differ (e.g. --dataset_backend h5 against a live
    --env_backend rlbench eval env)."""
    captured = {}

    def _infer_specs(req):
        captured["req"] = req
        return _box_spaces()

    _register_fake_backend(monkeypatch, "_test_fake", infer_specs=_infer_specs)

    args = _args(dataset_backend="_test_fake")
    args._test_fake = "matched-by-name"
    args.env_backend = "some_other_backend"

    infer_offline_dataset_specs(args)

    assert captured["req"].backend_config == "matched-by-name"


def test_infer_specs_routes_to_h5_with_action_bounds(monkeypatch):
    obs_space, action_space = _box_spaces()
    called = {}

    def _fake_infer_specs_from_h5(path, *, action_low, action_high):
        called["path"] = path
        called["action_low"] = action_low
        called["action_high"] = action_high
        return obs_space, action_space

    monkeypatch.setattr(
        "rl_garden.buffers.h5_dataset.infer_specs_from_h5", _fake_infer_specs_from_h5
    )

    result = infer_offline_dataset_specs(_args(dataset_backend="h5"))
    assert result == (obs_space, action_space)
    assert called == {"path": "demos/pickcube.h5", "action_low": -1.0, "action_high": 1.0}


def test_infer_specs_rejects_discrete_minari_action_space(monkeypatch):
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    discrete_action_space = spaces.Discrete(4)

    monkeypatch.setattr(
        "rl_garden.buffers.minari_dataset.infer_specs_from_minari",
        lambda dataset_id: (obs_space, discrete_action_space),
    )

    args = _args(dataset_backend="minari", offline_dataset="atari/pong/expert-v0")
    with pytest.raises(ValueError, match="Discrete"):
        infer_offline_dataset_specs(args)


def test_load_offline_dataset_routes_to_minari_with_num_episodes(monkeypatch):
    called = {}

    def _fake_load_minari(
        replay_buffer, dataset_id, *, num_episodes, reward_scale, reward_bias, success_key
    ):
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
        "rl_garden.buffers.minari_dataset.load_minari_dataset_to_replay_buffer",
        _fake_load_minari,
    )

    buffer = object()
    args = _args(
        dataset_backend="minari",
        offline_dataset="D4RL/halfcheetah/medium-v0",
        offline_num_traj=10,
    )
    loaded = load_offline_dataset(buffer, args)
    assert loaded == 7
    assert called["dataset_id"] == "D4RL/halfcheetah/medium-v0"
    assert called["num_episodes"] == 10


def test_load_offline_dataset_routes_to_d4rl_legacy_with_num_episodes(monkeypatch):
    called = {}

    def _fake_load(buffer, env_id, **kwargs):
        called.update(buffer=buffer, env_id=env_id, **kwargs)
        return 9

    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset.load_d4rl_legacy_dataset_to_replay_buffer",
        _fake_load,
    )
    buffer = object()
    loaded = load_offline_dataset(
        buffer,
        _args(
            dataset_backend="d4rl_legacy",
            offline_dataset="antmaze-test-v2",
            offline_num_traj=12,
        ),
    )

    assert loaded == 9
    assert called["buffer"] is buffer
    assert called["env_id"] == "antmaze-test-v2"
    assert called["num_episodes"] == 12


def test_infer_specs_routes_to_rlbench_with_obs_mode_and_cameras(monkeypatch):
    """Regression test for the found bug: --obs_mode/--rlbench.cameras
    weren't reaching rlbench's dataset spec inference/loading at all."""
    obs_space, action_space = _box_spaces()
    called = {}

    def _fake_infer_specs_from_rlbench(path, *, obs_mode, cameras, image_size):
        called["path"] = path
        called["obs_mode"] = obs_mode
        called["cameras"] = cameras
        called["image_size"] = image_size
        return obs_space, action_space

    monkeypatch.setattr(
        "rl_garden.buffers.rlbench_dataset.infer_specs_from_rlbench",
        _fake_infer_specs_from_rlbench,
    )

    args = _args(dataset_backend="rlbench", offline_dataset="/data/reach_target")
    args.obs_mode = "rgb"
    args.rlbench = SimpleNamespace(cameras=("front",), image_size=(64, 64))

    result = infer_offline_dataset_specs(args)
    assert result == (obs_space, action_space)
    assert called["obs_mode"] == "rgb"
    assert called["cameras"] == ("front",)
    assert called["image_size"] == (64, 64)


def test_infer_specs_unsupported_backend_raises():
    with pytest.raises(ValueError, match="Unknown dataset backend"):
        infer_offline_dataset_specs(_args(dataset_backend="bogus"))


def test_load_offline_dataset_unsupported_backend_raises():
    with pytest.raises(ValueError, match="Unknown dataset backend"):
        load_offline_dataset(object(), _args(dataset_backend="bogus"))
