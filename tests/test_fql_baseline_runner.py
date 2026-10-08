from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.fql import run_ogbench
from baselines.fql.run_ogbench import DEFAULT_FQL_SOURCE, FQL_REFERENCE_REVISION, PRESETS, build_command, inspect_source


def test_presets_cover_trained_task_phase_recipes_with_explicit_task_parameters():
    assert set(PRESETS) == {
        "cube-single-offline",
        "cube-double-off2on",
        "scene-off2on",
        "antmaze-large-task1-offline",
        "humanoidmaze-medium-task2-offline",
        "antsoccer-arena-task4-off2on",
        "puzzle-4x4-task4-off2on",
    }
    assert PRESETS["antmaze-large-task1-offline"].q_agg == "min"
    assert PRESETS["humanoidmaze-medium-task2-offline"].discount == pytest.approx(0.995)
    assert PRESETS["puzzle-4x4-task4-off2on"].reward_task_id == 4


def test_command_passes_recipe_parameters_instead_of_upstream_defaults(tmp_path: Path):
    preset = PRESETS["antmaze-large-task1-offline"]
    command = build_command(preset, seed=2, save_dir=tmp_path, run_group="group", eval_interval=100_000, save_interval=100_000)
    assert "--env_name=antmaze-large-navigate-singletask-task1-v0" in command
    assert "--agent.discount=0.99" in command
    assert "--agent.q_agg=min" in command
    assert "--agent.alpha=10" in command
    assert "--agent=agents/fql.py" in command
    assert "--online_steps=0" in command


def test_command_keeps_official_null_encoder_and_float32_recipe_defaults(tmp_path: Path):
    command = build_command(
        PRESETS["antsoccer-arena-task4-off2on"], seed=0, save_dir=tmp_path,
        run_group="group", eval_interval=100_000, save_interval=100_000,
    )
    assert "--agent.encoder=None" not in command
    assert "--agent.layer_norm=True" in command
    assert "--agent.actor_layer_norm=False" in command
    assert "--agent.actor_hidden_dims=(512,512,512,512)" in command
    assert "--agent.value_hidden_dims=(512,512,512,512)" in command


def test_missing_source_explains_required_pinned_revision(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=FQL_REFERENCE_REVISION):
        inspect_source(tmp_path / "missing")


def test_default_source_is_the_repository_reference_clone():
    assert DEFAULT_FQL_SOURCE.as_posix().endswith("3rd_party/fql")


def test_unverified_source_snapshot_warns_and_records_revision_and_dirty_state(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "main.py").touch()
    monkeypatch.setattr(run_ogbench, "_git_output", lambda _source, *args: (0, str(tmp_path) if args[-1] == "--show-toplevel" else "different-revision" if args[0] == "rev-parse" else "local-change"))
    metadata = inspect_source(tmp_path)
    assert metadata == {
        "path": str(tmp_path),
        "expected_revision": FQL_REFERENCE_REVISION,
        "revision": "different-revision",
        "dirty": "local-change",
        "verified": False,
        "warnings": [
            f"source revision differs from reference {FQL_REFERENCE_REVISION}: different-revision",
            "source worktree has local changes",
        ],
    }
    assert "WARNING: FQL source provenance" in capsys.readouterr().err


def test_dry_run_skips_source_dependency_gpu_and_output_creation(tmp_path: Path, monkeypatch, capsys):
    missing_source = tmp_path / "not-a-source"
    output_dir = tmp_path / "no-output"
    monkeypatch.setattr(run_ogbench, "inspect_source", lambda *_args, **_kwargs: pytest.fail("source inspected"))
    monkeypatch.setattr(run_ogbench, "_dependency_preflight", lambda: pytest.fail("dependencies inspected"))
    monkeypatch.setattr(run_ogbench, "_gpu_preflight", lambda: pytest.fail("GPU inspected"))
    assert run_ogbench.main([
        "--fql-source", str(missing_source), "--preset", "cube-single-offline", "--seed", "0",
        "--output-dir", str(output_dir), "--dry-run",
    ]) == 0
    assert not output_dir.exists()
    assert json.loads(capsys.readouterr().out)["cwd"] == str(missing_source)


def test_budget_override_is_effective_in_dry_run_without_source(tmp_path: Path, capsys):
    assert run_ogbench.main([
        "--preset", "cube-double-off2on", "--seed", "0", "--output-dir", str(tmp_path / "unused"),
        "--offline-steps", "7", "--online-steps", "3", "--dry-run",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["preset"]["offline_steps"] == 7
    assert printed["preset"]["online_steps"] == 3
    assert "--offline_steps=7" in printed["argv"]
    assert "--online_steps=3" in printed["argv"]


@pytest.mark.parametrize("flag", ("--offline-steps", "--online-steps"))
def test_negative_budget_is_rejected_before_any_launch(tmp_path: Path, flag: str):
    with pytest.raises(ValueError, match="nonnegative"):
        run_ogbench.main([
            "--preset", "cube-single-offline", "--seed", "0", "--output-dir", str(tmp_path / "unused"),
            flag, "-1", "--dry-run",
        ])


def test_dependency_preflight_requires_optax_and_setup_guidance(monkeypatch):
    versions = {name: "1" for name in ("jax", "jaxlib", "flax", "optax", "distrax", "ml-collections", "wandb")}
    versions.update({"ogbench": "1.1.0", "gymnasium": "0.29.1", "optax": None})
    monkeypatch.setattr(run_ogbench, "package_versions", lambda: versions)
    with pytest.raises(RuntimeError, match="optax.*fql-ogbench-reproduction"):
        run_ogbench._dependency_preflight()


def test_gpu_preflight_rejects_cpu_only_jax(monkeypatch):
    monkeypatch.setattr(
        run_ogbench.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stdout='[{"platform": "cpu", "id": 0}]', stderr=""),
    )
    with pytest.raises(RuntimeError, match="did not report a CUDA GPU"):
        run_ogbench._gpu_preflight()


def test_real_launch_writes_metadata_and_return_code(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_ogbench, "inspect_source", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(run_ogbench, "_dependency_preflight", lambda: {"ogbench": "1.1.0"})
    monkeypatch.setattr(run_ogbench, "_gpu_preflight", lambda: {"jax_devices": [{"platform": "gpu", "id": 0}]})
    calls: list[tuple[object, object]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(run_ogbench.subprocess, "run", fake_run)
    output_dir = tmp_path / "formal-output"
    assert run_ogbench.main([
        "--fql-source", str(tmp_path / "source"), "--preset", "cube-single-offline", "--seed", "1",
        "--output-dir", str(output_dir),
    ]) == 17
    assert calls[0][1]["cwd"] == (tmp_path / "source").resolve()
    assert json.loads((output_dir / "command.json").read_text())["official_config_defaults"]["agent.encoder"] is None
    assert json.loads((output_dir / "result.json").read_text()) == {"returncode": 17}
