"""Isaac Lab generation detection and the 2.x preset rejection (no Isaac Lab needed)."""
from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest

from rl_garden.envs.isaaclab import detect
from rl_garden.envs.isaaclab.v2.runtime import (
    _reject_camera_off_primary_gpu,
    _reject_presets,
)


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
