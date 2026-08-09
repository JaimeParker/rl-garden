"""Tests for offline/_runner.py's resource lifecycle."""

from types import SimpleNamespace

import pytest
from gymnasium import spaces

from rl_garden.training.offline import _runner
from rl_garden.training.offline.bc import BCArgs


@pytest.fixture(autouse=True)
def _disable_contract_validation_for_runner_test_doubles(monkeypatch):
    from rl_garden.training import inspection

    monkeypatch.setattr(
        inspection, "validate_constructor_coverage", lambda *args, **kwargs: None
    )


def test_run_offline_closes_resources_when_builder_fails(monkeypatch, tmp_path):
    closed = {"eval_env": False}

    def fake_infer_offline_dataset_specs(args):
        del args
        return (
            spaces.Box(-1, 1, shape=(3,), dtype="float32"),
            spaces.Box(-1, 1, shape=(2,), dtype="float32"),
        )

    def fake_should_create_eval_env(args):
        del args
        return True

    def fake_make_evaluation_env(backend_name, req):
        del backend_name, req
        return SimpleNamespace(close=lambda: closed.__setitem__("eval_env", True))

    monkeypatch.setattr(
        _runner, "infer_offline_dataset_specs", fake_infer_offline_dataset_specs
    )
    monkeypatch.setattr(_runner, "should_create_eval_env", fake_should_create_eval_env)
    monkeypatch.setattr(_runner, "make_evaluation_env", fake_make_evaluation_env)

    args = BCArgs(
        log_type="none",
        log_dir=str(tmp_path),
        offline_dataset_path="dummy_dataset",
        num_offline_steps=1,
        env_id="dummy-env",
    )

    def failing_builder(*unused):
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        _runner.run_offline(args, build_agent=failing_builder)

    assert closed == {"eval_env": True}


def test_run_offline_loads_checkpoint_last_and_respects_load_replay_buffer(
    monkeypatch, tmp_path
):
    events: list[object] = []

    class FakeAgent:
        device = "cpu"
        replay_buffer = object()

        def fit_obs_normalizer(self):
            events.append("fit_obs_normalizer")

        def load(self, path, *, load_replay_buffer):
            events.append(("load", path, load_replay_buffer))

    def fake_infer_offline_dataset_specs(args):
        del args
        return (
            spaces.Box(-1, 1, shape=(3,), dtype="float32"),
            spaces.Box(-1, 1, shape=(2,), dtype="float32"),
        )

    def fake_load_offline_dataset(replay_buffer, args):
        del replay_buffer, args
        events.append("load_offline_dataset")
        return 0

    def fake_run_offline_pretraining(*unused_args, **unused_kwargs):
        events.append("run_offline_pretraining")

    monkeypatch.setattr(
        _runner, "infer_offline_dataset_specs", fake_infer_offline_dataset_specs
    )
    monkeypatch.setattr(_runner, "load_offline_dataset", fake_load_offline_dataset)
    monkeypatch.setattr(
        _runner, "run_offline_pretraining", fake_run_offline_pretraining
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    args = BCArgs(
        log_type="none",
        log_dir=str(tmp_path),
        offline_dataset_path="dummy_dataset",
        num_offline_steps=1,
        env_id=None,
        load_checkpoint=str(checkpoint_path),
        load_replay_buffer=True,
    )

    def build_fake_agent(*unused):
        from rl_garden.training.inspection import construct_agent

        return construct_agent(FakeAgent)

    _runner.run_offline(args, build_agent=build_fake_agent)

    assert events == [
        "load_offline_dataset",
        "fit_obs_normalizer",
        ("load", str(checkpoint_path), True),
        "run_offline_pretraining",
    ]


def test_run_offline_respects_load_replay_buffer_false(monkeypatch, tmp_path):
    calls: list[tuple[str, bool]] = []

    class FakeAgent:
        device = "cpu"
        replay_buffer = object()

        def fit_obs_normalizer(self):
            pass

        def load(self, path, *, load_replay_buffer):
            calls.append((path, load_replay_buffer))

    monkeypatch.setattr(
        _runner,
        "infer_offline_dataset_specs",
        lambda args: (
            spaces.Box(-1, 1, shape=(3,), dtype="float32"),
            spaces.Box(-1, 1, shape=(2,), dtype="float32"),
        ),
    )
    monkeypatch.setattr(
        _runner, "load_offline_dataset", lambda replay_buffer, args: 0
    )
    monkeypatch.setattr(
        _runner, "run_offline_pretraining", lambda *unused, **unused_kw: None
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    args = BCArgs(
        log_type="none",
        log_dir=str(tmp_path),
        offline_dataset_path="dummy_dataset",
        num_offline_steps=1,
        env_id=None,
        load_checkpoint=str(checkpoint_path),
        load_replay_buffer=False,
    )

    def build_fake_agent(*unused):
        from rl_garden.training.inspection import construct_agent

        return construct_agent(FakeAgent)

    _runner.run_offline(args, build_agent=build_fake_agent)

    assert calls == [(str(checkpoint_path), False)]


def test_run_offline_dry_run_loads_checkpoint_without_replay_buffer(
    monkeypatch, tmp_path
):
    from rl_garden.training.offline import bc as bc_module
    from rl_garden.training.offline import registry as offline_registry

    calls: list[tuple[str, bool]] = []

    class FakeAgent:
        device = "cpu"

        def load(self, path, *, load_replay_buffer):
            calls.append((path, load_replay_buffer))

    def fake_infer_offline_dataset_specs(args):
        del args
        return (
            spaces.Box(-1, 1, shape=(3,), dtype="float32"),
            spaces.Box(-1, 1, shape=(2,), dtype="float32"),
        )

    def fake_build_bc(args, env_spec, logger, eval_env=None):
        from rl_garden.training.inspection import construct_agent

        del args, env_spec, logger, eval_env
        return construct_agent(FakeAgent)

    monkeypatch.setattr(
        _runner, "infer_offline_dataset_specs", fake_infer_offline_dataset_specs
    )
    monkeypatch.setattr(bc_module, "build_bc", fake_build_bc)

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    offline_registry.run_cli(
        [
            "bc",
            "--dry-run",
            "--offline-dataset-path",
            "dummy_dataset",
            "--num-offline-steps",
            "1",
            "--load-checkpoint",
            str(checkpoint_path),
            "--load-replay-buffer",
            "--log-type",
            "none",
            "--log-dir",
            str(tmp_path),
        ]
    )

    assert calls == [(str(checkpoint_path), False)]
