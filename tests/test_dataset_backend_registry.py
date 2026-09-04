"""Tests for the dataset-backend registry itself (rl_garden.buffers
.dataset_backend_registry) -- the mechanism, not any particular backend's
own loading logic (see test_{ogbench,rlbench,...}_dataset.py for those)."""
from __future__ import annotations

import pytest

from rl_garden.buffers.dataset_backend_registry import (
    DatasetBackend,
    DatasetRequest,
    infer_dataset_specs,
    load_dataset,
    register_dataset_backend,
)


def _unique_name() -> str:
    _unique_name.counter += 1
    return f"_test_backend_{_unique_name.counter}"


_unique_name.counter = 0


def test_register_dataset_backend_rejects_duplicate_name(monkeypatch):
    name = _unique_name()

    class _A(DatasetBackend):
        @classmethod
        def infer_specs(cls, req):
            return None

        @classmethod
        def load(cls, buffer, req):
            return 0

    register_dataset_backend(name, _A)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_dataset_backend(name, _A)
    finally:
        from rl_garden.buffers import dataset_backend_registry

        del dataset_backend_registry._REGISTRY[name]


def test_register_dataset_backend_rejects_missing_methods():
    name = _unique_name()

    class _MissingLoad(DatasetBackend):
        @classmethod
        def infer_specs(cls, req):
            return None

    with pytest.raises(TypeError, match="must implement"):
        register_dataset_backend(name, _MissingLoad)


def test_infer_dataset_specs_dispatches_to_registered_backend(monkeypatch):
    name = _unique_name()
    captured = {}

    class _Fake(DatasetBackend):
        @classmethod
        def infer_specs(cls, req):
            captured["req"] = req
            return "obs_space", "action_space"

        @classmethod
        def load(cls, buffer, req):
            return 0

    from rl_garden.buffers import dataset_backend_registry

    monkeypatch.setitem(dataset_backend_registry._REGISTRY, name, _Fake)

    req = DatasetRequest(path="/some/path", num_traj=5)
    result = infer_dataset_specs(req, backend_name=name)

    assert result == ("obs_space", "action_space")
    assert captured["req"] is req


def test_load_dataset_dispatches_to_registered_backend(monkeypatch):
    name = _unique_name()
    captured = {}

    class _Fake(DatasetBackend):
        @classmethod
        def infer_specs(cls, req):
            return None

        @classmethod
        def load(cls, buffer, req):
            captured["buffer"] = buffer
            captured["req"] = req
            return 123

    from rl_garden.buffers import dataset_backend_registry

    monkeypatch.setitem(dataset_backend_registry._REGISTRY, name, _Fake)

    buffer = object()
    req = DatasetRequest(path="/some/path")
    loaded = load_dataset(buffer, req, backend_name=name)

    assert loaded == 123
    assert captured["buffer"] is buffer
    assert captured["req"] is req


def test_unknown_backend_name_raises_with_available_backends_listed():
    with pytest.raises(ValueError, match="Unknown dataset backend 'bogus'. Available:"):
        infer_dataset_specs(DatasetRequest(path="x"), backend_name="bogus")


def test_every_shipped_backend_is_registered():
    import rl_garden.buffers  # noqa: F401 -- triggers registration.
    from rl_garden.buffers import dataset_backend_registry

    for name in ("h5", "minari", "d4rl_legacy", "robomimic", "ogbench", "rlbench", "metaworld"):
        assert name in dataset_backend_registry._REGISTRY
