"""Isaac Lab generation detection, the 2.x preset/final_obs rejection and the
3.0 compute_final_obs resolution (no Isaac Lab needed)."""
from __future__ import annotations

import importlib.metadata
import importlib.util
from types import SimpleNamespace

import pytest

from rl_garden.envs.isaaclab import detect
from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig
from rl_garden.envs.isaaclab.v2.runtime import (
    _reject_camera_off_primary_gpu,
    _reject_final_obs,
    _reject_presets,
)
from rl_garden.envs.isaaclab.v3.runtime import _apply_env_kwargs_and_final_obs


def _fake_find_spec(installed: set[str]):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in ("isaaclab", "isaaclab_physx"):
            return object() if name in installed else None
        return real_find_spec(name, *args, **kwargs)

    return find_spec


@pytest.mark.parametrize(
    ("installed", "expected"),
    [({"isaaclab", "isaaclab_physx"}, "v3"), ({"isaaclab"}, "v2")],
)
def test_generation_from_installed_packages(monkeypatch, installed, expected) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(installed))
    monkeypatch.setattr(detect, "isaaclab_version", lambda: "2.3.2")
    assert detect.generation() == expected


def test_generation_without_isaaclab_raises(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(set()))
    with pytest.raises(ImportError, match="isaaclab"):
        detect.generation()


def test_isaaclab_version_without_metadata_is_question_mark(monkeypatch) -> None:
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert detect.isaaclab_version() == "?"


def test_v2_accepts_no_presets() -> None:
    _reject_presets([], "2.3.2")


def test_v2_rejects_presets_with_detected_version() -> None:
    with pytest.raises(ValueError, match=r"presets need Isaac Lab 3\.0, detected 2\.x 2\.3\.2"):
        _reject_presets(["renderer=newton_renderer"], "2.3.2")


def test_3x_version_without_physx_raises(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"isaaclab"}))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.0.0")
    with pytest.raises(ImportError, match="isaaclab_physx"):
        detect.generation()


@pytest.mark.parametrize("version", ["0.54.2", "2.3.2", "?"])
def test_2x_or_unknown_version_without_physx_is_v2(monkeypatch, version) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"isaaclab"}))
    monkeypatch.setattr(detect, "isaaclab_version", lambda: version)
    assert detect.generation() == "v2"


@pytest.mark.parametrize(("is_visual", "sim_device"), [(False, "cuda:2"), (True, "cuda:0"), (True, "cuda")])
def test_v2_accepts_state_anywhere_and_camera_on_cuda0(is_visual, sim_device) -> None:
    _reject_camera_off_primary_gpu(is_visual, sim_device)


def test_v2_rejects_camera_off_cuda0() -> None:
    with pytest.raises(ValueError, match="cuda:0"):
        _reject_camera_off_primary_gpu(True, "cuda:2")


def _isaaclab_cfg(final_obs, env_kwargs=None) -> IsaacLabEnvConfig:
    return IsaacLabEnvConfig(
        env_id="RlGarden-Cartpole-Direct-v0",
        num_envs=4,
        seed=0,
        env_kwargs=env_kwargs or {},
        final_obs=final_obs,
    )


@pytest.mark.parametrize("final_obs", [None, False])
def test_v2_accepts_final_obs_unset_or_off(final_obs) -> None:
    _reject_final_obs(final_obs, "2.3.2")


def test_v2_make_env_rejects_final_obs_true_before_kit_boots(monkeypatch) -> None:
    from rl_garden.envs.isaaclab import env as isaaclab_env
    from rl_garden.envs.isaaclab.v2 import app as v2_app

    def kit_must_not_boot(*args, **kwargs):
        raise AssertionError("Kit booted before final_obs was rejected")

    monkeypatch.setattr(isaaclab_env, "generation", lambda: "v2")
    monkeypatch.setattr(detect, "isaaclab_version", lambda: "2.3.2")
    monkeypatch.setattr(v2_app, "get_or_launch_app", kit_must_not_boot)
    with pytest.raises(ValueError, match=r"final_obs True needs Isaac Lab 3\.0, detected 2\.x 2\.3\.2"):
        isaaclab_env.make_isaaclab_env(_isaaclab_cfg(True))


@pytest.mark.parametrize(("final_obs", "expected"), [(None, True), (True, True), (False, False)])
def test_v3_sets_compute_final_obs_after_env_kwargs(final_obs, expected) -> None:
    env_cfg = SimpleNamespace(compute_final_obs=False, episode_length_s=5.0)

    _apply_env_kwargs_and_final_obs(env_cfg, _isaaclab_cfg(final_obs, {"episode_length_s": 2.0}))

    assert env_cfg.compute_final_obs is expected
    assert env_cfg.episode_length_s == 2.0


def test_v3_rejects_compute_final_obs_in_env_kwargs() -> None:
    env_cfg = SimpleNamespace(compute_final_obs=False)

    with pytest.raises(ValueError, match="--isaaclab.final_obs"):
        _apply_env_kwargs_and_final_obs(env_cfg, _isaaclab_cfg(None, {"compute_final_obs": True}))


@pytest.mark.parametrize("final_obs", [None, True])
def test_v3_cfg_without_compute_final_obs_field_raises(final_obs) -> None:
    with pytest.raises(ValueError, match="no 'compute_final_obs' field"):
        _apply_env_kwargs_and_final_obs(SimpleNamespace(), _isaaclab_cfg(final_obs))


def test_v3_cfg_without_compute_final_obs_field_is_left_alone_when_off() -> None:
    env_cfg = SimpleNamespace()

    _apply_env_kwargs_and_final_obs(env_cfg, _isaaclab_cfg(False))

    assert not hasattr(env_cfg, "compute_final_obs")
