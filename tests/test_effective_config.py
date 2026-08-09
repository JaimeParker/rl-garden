from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rl_garden.common.effective_config import (
    ConfigError,
    EffectiveConfig,
    FieldSource,
    apply_strict_mapping,
    effective_config_json,
    load_preset,
    override_sources,
    persist_effective_config,
)


@dataclass
class _Backend:
    device: str = "cpu"
    sizes: list[int] = field(default_factory=lambda: [1])


@dataclass
class _BaseArgs:
    gamma: float = 0.8


@dataclass
class _Args(_BaseArgs):
    gamma: float = 0.9
    count: int = 1
    backend: _Backend = field(default_factory=_Backend)
    output: Path = Path("runs")


def _config(status="preflight") -> EffectiveConfig:
    return EffectiveConfig(
        schema_version=3,
        status=status,
        selection={"training_phase": "online", "algorithm": "sac"},
        inputs={"gamma": 0.9},
        active_environment={},
        algorithm={},
        derived={},
        sources={},
        runtime={},
    )


def test_strict_mapping_applies_nested_typed_values():
    args = _Args()

    apply_strict_mapping(
        args,
        {"gamma": 1, "backend": {"device": "cuda", "sizes": [2, 3]}, "output": "x"},
    )

    assert args.gamma == 1.0
    assert args.backend == _Backend(device="cuda", sizes=[2, 3])
    assert args.output == Path("x")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"missing": 1}, "Unknown preset field 'missing'"),
        ({"gamma": "fast"}, "Preset field 'gamma' must have type 'float'"),
        ({"backend": []}, "Preset field 'backend' must be a mapping"),
        ({"count": True}, "Preset field 'count' must have type 'int'"),
    ],
)
def test_strict_mapping_rejects_invalid_values(values, message):
    with pytest.raises(ConfigError, match=message):
        apply_strict_mapping(_Args(), values)


def test_load_preset_rejects_algorithm_identity(tmp_path):
    path = tmp_path / "preset.yaml"
    path.write_text("algorithm: sac\ngamma: 0.9\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="select phase and algorithm on the CLI"):
        load_preset(path)


def test_load_preset_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "preset.yaml"
    path.write_text("gamma: 0.8\ngamma: 0.9\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate key 'gamma'"):
        load_preset(path)


def test_sources_only_track_the_last_explicit_override():
    sources: dict[str, FieldSource] = {}
    override_sources(
        sources,
        {"gamma"},
        kind="preset",
        detail="preset.yaml",
    )
    override_sources(sources, {"gamma"}, kind="CLI", detail="argv")

    assert sources == {"gamma": FieldSource(kind="CLI", detail="argv")}


def test_effective_config_v3_serializes_and_atomically_replaces(tmp_path):
    path = tmp_path / "run" / "config.json"
    persist_effective_config(_config(), path)
    persist_effective_config(
        _config().materialized(
            active_environment={"backend": "fake"},
            algorithm={"target": "Fake"},
            derived={"steps": 3},
            runtime={"device": "cpu"},
        ),
        path,
    )

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 3
    assert payload["status"] == "materialized"
    assert "args" not in payload
    assert payload["runtime"]["device"] == "cpu"
    assert not (path.parent / ".config.json.tmp").exists()
    assert (
        json.loads(effective_config_json(_config()))["selection"]["algorithm"] == "sac"
    )

    with pytest.raises(TypeError):
        _config().inputs["gamma"] = 1.0


def test_json_value_uses_stable_type_for_runtime_objects():
    from rl_garden.common.effective_config import json_value

    class RuntimeObject:
        pass

    assert json_value(RuntimeObject()) == {
        "type": f"{__name__}.test_json_value_uses_stable_type_for_runtime_objects.<locals>.RuntimeObject"
    }


def test_wandb_v3_keeps_inputs_flat_and_materialized_details_namespaced():
    from rl_garden.common.logger import _flatten_wandb_config

    preflight = json.loads(effective_config_json(_config()))
    initial = _flatten_wandb_config(preflight)
    assert initial["gamma"] == 0.9
    assert initial["selection.algorithm"] == "sac"
    assert not any(key.startswith("effective.") for key in initial)

    materialized = json.loads(
        effective_config_json(
            _config().materialized(
                active_environment={"backend": "fake"},
                algorithm={"target": "Fake"},
                derived={"steps": 3},
                runtime={"device": "cpu"},
            )
        )
    )
    updated = _flatten_wandb_config(materialized)
    assert updated["gamma"] == 0.9
    assert updated["effective.algorithm.target"] == "Fake"
    assert updated["effective.runtime.device"] == "cpu"


def test_backend_registration_rejects_pre_v2_contract():
    from rl_garden.envs.backend_registry import EnvBackend, register_env_backend

    class LegacyBackend(EnvBackend):
        config_field = "legacy_test"

        @classmethod
        def make_train_env(cls, req):
            return None

        @classmethod
        def make_eval_env(cls, req):
            return None

    with pytest.raises(TypeError, match="api_version = 2"):
        register_env_backend("legacy_test", LegacyBackend)
