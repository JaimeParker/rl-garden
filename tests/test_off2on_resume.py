"""A resumed online stage of an offline-to-online run continues in the state it
was saved in, for every registered off2on algorithm and replay mode."""
from __future__ import annotations

import logging
import shutil

import h5py
import pytest
import torch

try:
    from tests._cli import run_entry
    from tests.test_entrypoint_builds import (
        PHASES, _registered, build_argv, cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from tests.test_replay_buffer_state import assert_tree_equal
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry
    from test_entrypoint_builds import (
        PHASES, _registered, build_argv, cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from test_replay_buffer_state import assert_tree_equal

from rl_garden.algorithms.off2on import offline_buffer_path_for_checkpoint
from rl_garden.buffers import ReplayBuffer, load_h5_dataset_to_replay_buffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer
from rl_garden.common.checkpoint import save_replay_buffer_file
from rl_garden.training import inspection

__all__ = ["cpu_only_and_restored_rng", "shared_inputs"]  # fixtures

ALGORITHMS = sorted(_registered("off2on"))
MODES = {
    "empty": {"online_replay_mode": "empty"},
    "append": {"online_replay_mode": "append"},
    "mixed": {"online_replay_mode": "mixed", "offline_data_ratio": "0.5"},
    "mixed-auto": {"online_replay_mode": "mixed", "offline_data_ratio": "auto"},
}
FAST = {
    ("wsrl", "empty"), ("calql", "mixed-auto"), ("iql", "append"),
    ("fac", "mixed"), ("value_flows", "empty"), ("spot", "mixed"),
}
CASES = [
    pytest.param(
        name, mode, id=f"{name}-{mode}", marks=() if (name, mode) in FAST else pytest.mark.slow
    )
    for name in ALGORITHMS
    for mode in MODES
]
#: Online stage: 16 env steps in 8-step episodes.
TRAINING = {
    "env_id": "rlg-entrypoint-build-halfcheetah-pendulum-v0",
    "training_freq": "4", "utd": "0.5", "num_steps": "8", "learning_starts": "0",
    "vae_iterations": "2", "bc_pretrain_steps": "2", "horizon_length": "2", "hidden_layers": "2",
    "num_samples": "2", "eval_freq": "0", "warmup_steps": "0",
}
#: Cal-QL keeps its MC buffer in "empty", which samples complete episodes only.
EMPTY_MC = {("calql", "empty"): {"learning_starts": "20"}}
OFFLINE_ONLINE = {"num_offline_steps": "2", "num_online_steps": "16", "save_replay_buffer": True}
CONTINUE_ONLY = {"num_offline_steps": "0", "num_online_steps": "0"}


def _inputs(name, shared, tmp_path, with_dataset: bool = True, **values):
    inputs = {"action_low": "-2", "action_high": "2"}
    if with_dataset:
        inputs = inputs_for("off2on", name, shared, tmp_path)
    return {**TRAINING, **inputs, **values}


def _run(name, log_dir, inputs, monkeypatch, *flags: str):
    """``(agent, CliResult)`` of a real ``train_off2on`` run."""
    built = []
    construct = inspection.construct_agent

    def capture(*args, **kwargs):
        built.append(construct(*args, **kwargs))
        return built[-1]

    argv = build_argv("off2on", name, log_dir, inputs=inputs)
    with monkeypatch.context() as patch:
        patch.setattr(inspection, "construct_agent", capture)
        result = run_entry(PHASES["off2on"][0], [*argv, *flags])
    return (built[-1] if built else None), result


def _trained(name, mode, shared, tmp_path, monkeypatch, **values):
    agent, result = _run(
        name,
        tmp_path / "first",
        _inputs(name, shared, tmp_path, 
                **{**MODES[mode], **EMPTY_MC.get((name, mode), {}), **OFFLINE_ONLINE, **values}),
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    (checkpoint,) = (tmp_path / "first").glob("*/checkpoints/final.pt")
    return agent, checkpoint


def _stage(agent) -> dict:
    offline = agent.offline_replay_buffer
    return {
        "online_start_step": agent._online_start_step,
        "online_replay_mode": agent.online_replay_mode,
        "offline_data_ratio": agent.offline_data_ratio,
        "online_buffer": (type(agent.replay_buffer), len(agent.replay_buffer)),
        "offline_buffer": None if offline is None else (type(offline), len(offline)),
        "initial_phase": (agent.initial_training_phase, agent._initial_phase_start_step),
        "eval_freq": agent.eval_freq,
        "gamma": agent.gamma,
        "global_step": agent._global_step,
        **{
            key: getattr(agent, key)
            for key in (
                "use_cql_loss", "cql_alpha", "_online_finetuning", "policy_extraction",
                "_spot_online_update_start",
            )
            if hasattr(agent, key)
        },
    }


@pytest.mark.parametrize("name,mode", CASES)
def test_online_stage_survives_a_resume(name, mode, tmp_path, shared_inputs, monkeypatch):  # noqa: F811
    agent, checkpoint = _trained(
        name, mode, shared_inputs, tmp_path, monkeypatch, warmup_steps="6"
    )
    saved = _stage(agent)
    assert saved["online_replay_mode"] == MODES[mode]["online_replay_mode"]

    resumed, result = _run(
        name,
        tmp_path / "second",
        _inputs(
            name, shared_inputs, tmp_path, with_dataset=False,
            **MODES[mode], **CONTINUE_ONLY, warmup_steps="6",
        ),
        monkeypatch,
        "--load-checkpoint", str(checkpoint), "--load-replay-buffer",
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert _stage(resumed) == saved
    for buffer, restored in (
        (agent.replay_buffer, resumed.replay_buffer),
        (agent.offline_replay_buffer, resumed.offline_replay_buffer),
    ):
        if buffer is not None:
            assert_tree_equal(buffer.state_dict(), restored.state_dict())

    # It keeps training, and what it saves is resumable again.
    target = agent._global_step + 8
    resumed.learn(target)
    path = resumed.save(tmp_path / "again.pt", include_replay_buffer=True)
    second = _rebuild(name, tmp_path, shared_inputs, monkeypatch, mode)
    second.load(path)
    expected = _stage(resumed)
    assert _stage(second) == expected


def _rebuild(name, tmp_path, shared, monkeypatch, mode):
    agent, result = _run(
        name,
        tmp_path / "third",
        _inputs(name, shared, tmp_path, with_dataset=False, **MODES[mode], warmup_steps="6"),
        monkeypatch,
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return agent


# --- offline data of the mixed stage -----------------------------------------


def _resume(name, checkpoint, tmp_path, shared, monkeypatch, *flags, dataset=False, **values):
    return _run(
        name,
        tmp_path / "resumed",
        _inputs(
            name, shared, tmp_path, with_dataset=dataset, **{**CONTINUE_ONLY, **values}
        ),
        monkeypatch,
        "--load-checkpoint", str(checkpoint), *flags,
    )


def test_mixed_reloads_the_dataset_into_the_offline_buffer(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    resumed, result = _resume(
        "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, dataset=True, **MODES["mixed"],
    )
    assert result.returncode == 0, result.stderr[-1500:]
    saved = _stage(agent)
    saved["online_buffer"] = (saved["online_buffer"][0], 0)
    assert _stage(resumed) == saved
    assert resumed._offline_probe_batch is not None


def test_saved_offline_buffer_is_preferred_to_the_dataset(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        resumed, result = _resume(
            "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
            dataset=True, **MODES["mixed"],
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "the offline buffer is as the checkpoint was saved with it" in caplog.text
    assert _stage(resumed) == _stage(agent)


def test_mixed_without_offline_data_fails(tmp_path, shared_inputs, monkeypatch):  # noqa: F811
    _, checkpoint = _trained("calql", "mixed-auto", shared_inputs, tmp_path, monkeypatch)
    _, result = _resume("calql", checkpoint, tmp_path, shared_inputs, monkeypatch)
    assert result.returncode == 1
    assert "--offline-dataset" in result.stderr and "mixed" in result.stderr


@pytest.mark.parametrize("name", ["awac", "wsrl"])
@pytest.mark.parametrize(
    "mode,restore_replay,reason",
    [
        ("empty", True, "discards offline data"),
        ("empty", False, "discards offline data"),
        ("append", True, "already holds the offline data"),
    ],
)
def test_dataset_the_saved_stage_does_not_load(
    name, mode, restore_replay, reason, tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent, checkpoint = _trained(name, mode, shared_inputs, tmp_path, monkeypatch)
    flags = ["--load-replay-buffer"] if restore_replay else []
    with caplog.at_level(logging.WARNING):
        resumed, result = _resume(
            name, checkpoint, tmp_path, shared_inputs, monkeypatch, *flags, dataset=True,
            **MODES[mode],
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "--offline-dataset is not loaded" in caplog.text and reason in caplog.text
    if restore_replay:
        assert _stage(resumed) == _stage(agent)
    else:
        assert len(resumed.replay_buffer) == 0
    assert resumed._offline_probe_batch is None
    assert_tree_equal(agent.policy.state_dict(), resumed.policy.state_dict())


@pytest.mark.parametrize("name", ["awac", "wsrl"])
@pytest.mark.parametrize("replay_file_missing", [False, True])
def test_append_reloads_the_dataset_when_its_buffer_was_not_restored(
    name, replay_file_missing, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    """The flag alone does not count as restored: a checkpoint saved without
    replay loads nothing."""
    _, checkpoint = _trained(
        name, "append", shared_inputs, tmp_path, monkeypatch,
        save_replay_buffer=not replay_file_missing,
    )
    flags = ["--load-replay-buffer"] if replay_file_missing else []
    resumed, result = _resume(
        name, checkpoint, tmp_path, shared_inputs, monkeypatch, *flags, dataset=True,
        **MODES["append"],
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(resumed.replay_buffer) == 48
    assert resumed._offline_probe_batch is not None


def test_append_without_its_buffer_or_a_dataset_is_warned_about(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    _, checkpoint = _trained("iql", "append", shared_inputs, tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        resumed, result = _resume(
            "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, **MODES["append"]
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "any offline data it held is gone" in caplog.text
    assert len(resumed.replay_buffer) == 0


def test_api_load_without_the_offline_buffer_refuses_to_train(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    agent = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "mixed")
    agent.load(checkpoint, load_replay_buffer=False)
    assert agent.lacks_offline_data and agent.offline_replay_buffer is None
    with pytest.raises(RuntimeError, match="no offline data"):
        agent.train(1)


# --- flags -------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,named",
    [
        (["--online-replay-mode", "append"], "--online-replay-mode append (saved: mixed)"),
        (["--offline-data-ratio", "0.25"], "--offline-data-ratio 0.25 (saved: 0.5)"),
        (["--no-online-use-cql-loss"], "--online-use-cql-loss False (saved: True)"),
        (["--warmup-steps", "3"], "--warmup-steps 3 (saved: 6)"),
        (["--num-offline-steps", "1"], "--num-offline-steps 0"),
    ],
)
def test_flags_contradicting_the_saved_stage_are_rejected(
    flags, named, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint = _trained(
        "calql", "mixed", shared_inputs, tmp_path, monkeypatch, warmup_steps="6"
    )
    inputs = _inputs("calql", shared_inputs, tmp_path, **CONTINUE_ONLY)
    if "--warmup-steps" not in flags:
        inputs["warmup_steps"] = "6"
    for flag in ("online_replay_mode", "offline_data_ratio"):
        inputs.pop(flag, None)
    _, result = _run(
        "calql", tmp_path / "resumed", inputs, monkeypatch,
        "--load-checkpoint", str(checkpoint), "--load-replay-buffer", *flags,
    )
    assert result.returncode == 1
    assert named in result.stderr


def test_defaults_differing_from_the_saved_stage_are_reported(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent, checkpoint = _trained(
        "wsrl", "mixed", shared_inputs, tmp_path, monkeypatch, warmup_steps="6"
    )
    inputs = _inputs("wsrl", shared_inputs, tmp_path, **CONTINUE_ONLY)
    del inputs["warmup_steps"]
    with caplog.at_level(logging.WARNING):
        resumed, result = _run(
            "wsrl", tmp_path / "resumed", inputs, monkeypatch,
            "--load-checkpoint", str(checkpoint), "--load-replay-buffer",
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "online_replay_mode=mixed, offline_data_ratio=0.5" in caplog.text
    assert "warmup_steps=6" in caplog.text
    assert _stage(resumed) == _stage(agent)


def test_eval_cadence_is_restored_and_an_explicit_one_wins(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained(
        "iql", "append", shared_inputs, tmp_path, monkeypatch,
        eval_freq="0", online_eval_freq="8", num_eval_steps="8",
    )
    assert agent.eval_freq == 8
    inputs = _inputs("iql", shared_inputs, tmp_path, with_dataset=False, **MODES["append"],
                     **CONTINUE_ONLY, num_eval_steps="8")
    flags = ["--load-checkpoint", str(checkpoint)]
    for values, expected in (({}, 8), ({"online_eval_freq": "4"}, 4), ({"eval_freq": "0"}, 0)):
        argv = build_argv("off2on", "iql", tmp_path / "resumed", inputs=inputs)
        position = argv.index("--eval-freq")
        del argv[position : position + 2]
        for key, value in values.items():
            argv += [f"--{key.replace('_', '-')}", value]
        built = []
        construct = inspection.construct_agent
        with monkeypatch.context() as patch:
            patch.setattr(
                inspection, "construct_agent",
                lambda *a, **k: built.append(construct(*a, **k)) or built[-1],
            )
            result = run_entry(PHASES["off2on"][0], [*argv, *flags])
        assert result.returncode == 0, result.stderr[-1500:]
        assert built[-1].eval_freq == expected, values


# --- offline-stage checkpoints -----------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ["wsrl", "calql", "iql"])
@pytest.mark.parametrize("more_offline_steps", ["0", "2"])
def test_offline_stage_checkpoints_start_a_new_online_stage(
    name, mode, more_offline_steps, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint = _trained(name, "append", shared_inputs, tmp_path, monkeypatch)
    resumed, result = _run(
        name,
        tmp_path / "resumed",
        _inputs(
            name, shared_inputs, tmp_path, **MODES[mode], **EMPTY_MC.get((name, mode), {}),
            num_offline_steps=more_offline_steps, num_online_steps="16",
        ),
        monkeypatch,
        "--load-checkpoint", str(checkpoint.with_name("offline_final.pt")),
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert resumed._online_start_step == 2 + int(more_offline_steps)
    expected = {"append": 48 + 16}.get(mode, 16)
    assert len(resumed.replay_buffer) == expected
    if mode.startswith("mixed"):
        assert len(resumed.offline_replay_buffer) == 48


def _rows(buffer) -> list[tuple]:
    size = buffer.pos if not buffer.full else buffer.per_env_buffer_size
    states = buffer.obs["state"][:size].reshape(-1, 3)
    return [tuple(row) for row in torch.cat([states, buffer.actions[:size].reshape(-1, 1)], 1).tolist()]


@pytest.mark.parametrize("more_offline_steps", ["0", "2"])
@pytest.mark.parametrize("mode", ["empty", "append", "mixed"])
@pytest.mark.parametrize("name", ["iql", "calql", "spot", "acfql"])
def test_offline_stage_restored_buffer_holds_each_row_once(
    name, mode, more_offline_steps, tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    trained, checkpoint = _trained(name, "append", shared_inputs, tmp_path, monkeypatch)
    seen = {}
    switch = type(trained).switch_to_online_mode

    def switch_recording(agent, *args, **kwargs):
        seen.update(rows=_rows(agent.replay_buffer), probe=agent._offline_probe_batch)
        if hasattr(agent.replay_buffer, "_externally_valid"):
            seen["mc"] = agent.replay_buffer.sampleable_size
        return switch(agent, *args, **kwargs)

    monkeypatch.setattr(type(trained), "switch_to_online_mode", switch_recording)
    with caplog.at_level(logging.WARNING):
        resumed, result = _run(
            name, tmp_path / "resumed",
            _inputs(
                name, shared_inputs, tmp_path, **MODES[mode],
                num_offline_steps=more_offline_steps, num_online_steps="0",
            ),
            monkeypatch,
            "--load-checkpoint", str(checkpoint.with_name("offline_final.pt")),
            "--load-replay-buffer",
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(seen["rows"]) == len(set(seen["rows"])) == 48
    assert sorted(seen["rows"]) == sorted(_rows(trained.replay_buffer)[:48])
    assert seen["probe"] is not None
    assert seen.get("mc", 48) == 48
    assert "--offline-dataset is not loaded" in caplog.text
    kept = {
        "empty": resumed.replay_buffer, "append": resumed.replay_buffer,
        "mixed": resumed.offline_replay_buffer,
    }[mode]
    assert len(kept) == (0 if mode == "empty" else 48)


def test_offline_stage_file_without_rows_takes_the_dataset(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "append")
    path = agent.save(tmp_path / "empty.pt", include_replay_buffer=True)
    resumed, result = _run(
        "iql", tmp_path / "resumed",
        _inputs("iql", shared_inputs, tmp_path, **MODES["append"],
                num_offline_steps="2", num_online_steps="0"),
        monkeypatch, "--load-checkpoint", str(path), "--load-replay-buffer",
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(resumed.replay_buffer) == 48


def test_offline_stage_warning_says_what_the_restored_buffer_holds(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    _, checkpoint = _trained("iql", "append", shared_inputs, tmp_path, monkeypatch)
    other = tmp_path / "other.h5"
    with h5py.File(shared_inputs.h5) as source, h5py.File(other, "w") as target:
        source.copy("traj_0", target)
    inputs = _inputs("iql", shared_inputs, tmp_path, **MODES["append"], **CONTINUE_ONLY)
    inputs["offline_dataset"] = str(other)
    with caplog.at_level(logging.WARNING):
        resumed, result = _run(
            "iql", tmp_path / "resumed", inputs, monkeypatch,
            "--load-checkpoint", str(checkpoint.with_name("offline_final.pt")),
            "--load-replay-buffer",
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(resumed.replay_buffer) == 48
    assert "48 transitions" in caplog.text and "without --load-replay-buffer" in caplog.text


# --- checkpoints that do not record the stage --------------------------------


def _forget_stage(checkpoint, *, run_config: bool):
    contents = torch.load(checkpoint)
    del contents["state"]["extra"]["online_stage"]
    if run_config:
        del contents["metadata"]["run_config"]
    torch.save(contents, checkpoint)


def test_unrecorded_stage_comes_from_the_runs_own_configuration(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("calql", "mixed-auto", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=False)
    resumed, result = _resume(
        "calql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        dataset=True,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert _stage(resumed) == _stage(agent)


def test_unrecorded_stage_needs_the_flag(tmp_path, shared_inputs, monkeypatch):  # noqa: F811
    agent, checkpoint = _trained("iql", "append", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=True)
    _, result = _resume(
        "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer"
    )
    assert result.returncode == 1
    assert "--online-replay-mode" in result.stderr

    resumed, result = _resume(
        "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        **MODES["append"],
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert _stage(resumed) == _stage(agent)


def test_unrecorded_mixed_stage_needs_the_ratio_too(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=True)
    _, result = _resume(
        "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        dataset=True, online_replay_mode="mixed",
    )
    assert result.returncode == 1
    assert "--offline-data-ratio" in result.stderr

    resumed, result = _resume(
        "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        dataset=True, **MODES["mixed"],
    )
    assert result.returncode == 0, result.stderr[-1500:]
    saved = _stage(agent)
    assert _stage(resumed) == saved


def test_unrecorded_empty_stage_is_read_off_its_plain_replay_file(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("wsrl", "empty", shared_inputs, tmp_path, monkeypatch)
    assert type(agent.replay_buffer) is ReplayBuffer
    _forget_stage(checkpoint, run_config=True)
    resumed, result = _resume(
        "wsrl", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert _stage(resumed) == _stage(agent)


def test_unrecorded_empty_stage_with_mc_replay_file_keeps_its_buffer(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("wsrl", "append", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=True)
    resumed, result = _resume(
        "wsrl", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        **MODES["empty"],
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert type(resumed.replay_buffer) is MCReplayBuffer
    assert len(resumed.replay_buffer) == len(agent.replay_buffer)


# --- what load() restored -----------------------------------------------------


def _online_agent(name, mode, tmp_path, shared, monkeypatch, *, offline_data_ratio=0.0):
    """An agent switched online with nothing in its buffers."""
    agent = _rebuild(name, tmp_path, shared, monkeypatch, mode)
    agent.switch_to_online_mode(mode, offline_data_ratio)
    return agent


def test_an_offline_file_of_another_save_is_not_loaded(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    agent.save(checkpoint)
    assert offline_buffer_path_for_checkpoint(checkpoint).exists()
    loaded = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "mixed")
    loaded.load(checkpoint)
    assert loaded.offline_replay_buffer is None
    assert loaded.lacks_offline_data


def test_a_recorded_offline_file_that_is_missing_raises(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    path = offline_buffer_path_for_checkpoint(checkpoint)
    path.unlink()
    loaded = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "mixed")
    with pytest.raises(FileNotFoundError, match=path.name):
        loaded.load(checkpoint)
    loaded.load(checkpoint, load_replay_buffer=False)
    assert loaded.lacks_offline_data


def test_an_offline_file_of_another_length_raises(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "mixed", shared_inputs, tmp_path, monkeypatch)
    save_replay_buffer_file(
        offline_buffer_path_for_checkpoint(checkpoint), agent.replay_buffer
    )
    loaded = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "mixed")
    with pytest.raises(ValueError, match="16 transitions.*48"):
        loaded.load(checkpoint)


@pytest.mark.parametrize("mode", ["append", "mixed"])
def test_a_restored_empty_buffer_is_not_refilled(
    mode, tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent = _online_agent(
        "iql", mode, tmp_path, shared_inputs, monkeypatch,
        offline_data_ratio="auto" if mode == "mixed" else 0.0,
    )
    path = agent.save(tmp_path / "empty.pt", include_replay_buffer=True)
    with caplog.at_level(logging.WARNING):
        resumed, result = _resume(
            "iql", path, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
            dataset=True, warmup_steps="6", **MODES["mixed-auto" if mode == "mixed" else mode],
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert "--offline-dataset is not loaded" in caplog.text
    kept = resumed.replay_buffer if mode == "append" else resumed.offline_replay_buffer
    assert len(kept) == 0


def test_a_mixed_stage_saved_without_offline_data_takes_a_dataset(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent = _online_agent(
        "iql", "mixed", tmp_path, shared_inputs, monkeypatch, offline_data_ratio="auto"
    )
    path = agent.save(tmp_path / "no_replay.pt")
    for dataset, offline in ((True, 48), (False, None)):
        resumed, result = _resume(
            "iql", path, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
            dataset=dataset, warmup_steps="6", **MODES["mixed-auto"],
        )
        assert result.returncode == 0, result.stderr[-1500:]
        buffer = resumed.offline_replay_buffer
        assert (None if buffer is None else len(buffer)) == offline


def test_mixed_at_ratio_zero_needs_no_offline_data(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint = _trained(
        "iql", "mixed", shared_inputs, tmp_path, monkeypatch,
        offline_data_ratio="0.0", save_replay_buffer=False,
    )
    for dataset, offline in ((False, None), (True, 48)):
        resumed, result = _resume(
            "iql", checkpoint, tmp_path, shared_inputs, monkeypatch, dataset=dataset,
            online_replay_mode="mixed", offline_data_ratio="0.0", num_online_steps="16",
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert len(resumed.replay_buffer) == 16
        buffer = resumed.offline_replay_buffer
        assert (None if buffer is None else len(buffer)) == offline
        resumed.train(1)


def test_a_replay_file_of_another_save_is_not_restored(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("iql", "append", shared_inputs, tmp_path, monkeypatch)
    agent.save(checkpoint)
    assert checkpoint.with_name("replay_buffer_final.pt").exists()
    loaded = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "append")
    loaded.load(checkpoint)
    assert not loaded.replay_buffer_restored
    assert len(loaded.replay_buffer) == 0


def test_api_load_of_a_stageless_checkpoint_waits_for_the_declared_mode(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    """The run configuration is the off2on runner's to read; the Python API
    leaves the mode to the caller."""
    agent, checkpoint = _trained("iql", "append", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=False)
    loaded = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "append")
    loaded.load(checkpoint)
    assert loaded.online_replay_mode is None and loaded.replay_buffer_restored
    loaded.declare_online_replay_mode("append")
    assert loaded.online_replay_mode == "append"
    assert len(loaded.replay_buffer) == len(agent.replay_buffer) > 0


def test_only_cal_ql_records_its_replay_buffer_class(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    for name, records in (("iql", False), ("wsrl", True)):
        _, checkpoint = _trained(name, "append", shared_inputs, tmp_path / name, monkeypatch)
        stage = torch.load(checkpoint)["state"]["extra"]["online_stage"]
        assert ("replay_buffer_class" in stage) == records, name


def test_a_configured_empty_stage_keeps_its_restored_plain_buffer(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    """A checkpoint with its run configuration and no stage: the runner states
    the mode after the plain replay file was read into a plain buffer."""
    agent, checkpoint = _trained("wsrl", "empty", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=False)
    resumed, result = _resume(
        "wsrl", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert type(resumed.replay_buffer) is ReplayBuffer
    assert resumed.online_replay_mode == "empty"
    assert_tree_equal(agent.replay_buffer.state_dict(), resumed.replay_buffer.state_dict())


def _stageless_checkpoint_and_agent_with_another_buffer(
    file_mode, agent_mode, tmp_path, shared, monkeypatch
):
    """A Cal-QL checkpoint of ``file_mode`` without its stage and run
    configuration, and a trained agent of ``agent_mode`` whose filled replay
    buffer is of the other class."""
    source, checkpoint = _trained("wsrl", file_mode, shared, tmp_path / "source", monkeypatch)
    _forget_stage(checkpoint, run_config=True)
    agent, _ = _trained("wsrl", agent_mode, shared, tmp_path / "agent", monkeypatch)
    assert type(agent.replay_buffer) is not type(source.replay_buffer)
    assert len(agent.replay_buffer) > 0
    return source, checkpoint, agent


@pytest.mark.parametrize("strict", [True, False])
def test_a_stageless_mc_replay_file_is_read_into_an_mc_buffer(
    strict, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    source, checkpoint, agent = _stageless_checkpoint_and_agent_with_another_buffer(
        "append", "empty", tmp_path, shared_inputs, monkeypatch
    )
    agent.load(checkpoint, strict=strict)
    assert type(agent.replay_buffer) is MCReplayBuffer
    assert_tree_equal(source.replay_buffer.state_dict(), agent.replay_buffer.state_dict())
    assert agent.online_replay_mode is None and agent.replay_buffer_restored
    restored = agent.replay_buffer
    agent.declare_online_replay_mode("append")
    assert agent.replay_buffer is restored


def test_a_stageless_plain_replay_file_is_read_into_a_plain_buffer(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    source, checkpoint, agent = _stageless_checkpoint_and_agent_with_another_buffer(
        "empty", "append", tmp_path, shared_inputs, monkeypatch
    )
    agent.load(checkpoint)
    assert type(agent.replay_buffer) is ReplayBuffer
    assert_tree_equal(source.replay_buffer.state_dict(), agent.replay_buffer.state_dict())
    # Only the empty mode leaves the plain buffer behind.
    assert agent.online_replay_mode == "empty"


def test_a_stageless_checkpoint_without_its_replay_file_takes_the_declared_modes_buffer(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint, agent = _stageless_checkpoint_and_agent_with_another_buffer(
        "append", "empty", tmp_path, shared_inputs, monkeypatch
    )
    agent.load(checkpoint, load_replay_buffer=False)
    assert agent.online_replay_mode is None and not agent.replay_buffer_restored
    agent.declare_online_replay_mode("append")
    assert type(agent.replay_buffer) is MCReplayBuffer
    assert len(agent.replay_buffer) == 0


def test_unrecorded_stage_is_inferred_from_the_replay_file_the_checkpoint_names(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, checkpoint = _trained("wsrl", "empty", shared_inputs, tmp_path, monkeypatch)
    _forget_stage(checkpoint, run_config=True)
    replay = checkpoint.with_name("replay_buffer_final.pt")
    replay.rename(checkpoint.with_name("moved.pt"))
    contents = torch.load(checkpoint)
    contents["metadata"]["replay_buffer_path"] = "moved.pt"
    torch.save(contents, checkpoint)
    resumed, result = _resume(
        "wsrl", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert _stage(resumed) == _stage(agent)


# --- algorithm state ----------------------------------------------------------


def test_spot_continues_its_cooling_and_discount(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    cooling = {**MODES["append"], "lambd_cool": True, "lambd_end": "0.2"}
    agent, checkpoint = _trained(
        "spot", "append", shared_inputs, tmp_path, monkeypatch,
        **cooling, online_discount="0.9",
    )
    assert 0 < agent._current_lambd() < agent.lambd
    with caplog.at_level(logging.WARNING):
        resumed, result = _run(
            "spot", tmp_path / "second",
            _inputs("spot", shared_inputs, tmp_path, with_dataset=False,
                    **{**CONTINUE_ONLY, **cooling, "num_online_steps": "1000"}),
            monkeypatch, "--load-checkpoint", str(checkpoint), "--load-replay-buffer", "--dry-run",
        )
    assert result.returncode == 0, result.stderr[-1500:]
    assert resumed.max_online_updates == agent.max_online_updates
    assert resumed._current_lambd() == agent._current_lambd()
    assert resumed.gamma == 0.9
    assert "online_discount=0.9" in caplog.text
    _, result = _resume(
        "spot", checkpoint, tmp_path, shared_inputs, monkeypatch, "--load-replay-buffer",
        **cooling, online_discount="0.8",
    )
    assert result.returncode == 1
    assert "--online-discount 0.8 (saved: 0.9)" in result.stderr


@pytest.mark.slow
def test_compiled_losses_follow_the_loaded_stage(tmp_path):
    """Loaded after its losses were traced with the offline CQL weight, an agent
    computes what one that was traced with the online weight only computes."""
    from test_checkpoint import _add_state_transitions, _state_env, _wsrl_kwargs

    from rl_garden.algorithms import WSRL

    kwargs = {
        **_wsrl_kwargs(), "online_use_cql_loss": True, "online_cql_alpha": 0.25, "use_compile": True,
    }
    agent = WSRL(env=_state_env(), **kwargs)
    _add_state_transitions(agent)
    agent.train(1)
    agent.switch_to_online_mode("append")
    path = agent.save(tmp_path / "online.pt", include_replay_buffer=True)

    loaded = WSRL(env=_state_env(), **kwargs)
    _add_state_transitions(loaded)
    loaded.train(1)
    loaded.load(path)
    reference = WSRL(env=_state_env(), **{**kwargs, "cql_alpha": 0.25})
    reference.policy.load_state_dict(loaded.policy.state_dict())
    offline_weight = WSRL(env=_state_env(), **kwargs)
    offline_weight.policy.load_state_dict(loaded.policy.state_dict())

    batch = loaded.replay_buffer.sample(4)
    losses, gradients = [], []
    for current in (loaded, reference, offline_weight):
        torch.manual_seed(0)
        loss, _ = current._critic_loss(batch)
        current.policy.zero_grad()
        loss.backward()
        losses.append(loss.detach())
        gradients.append([p.grad.clone() for p in current.policy.parameters() if p.grad is not None])
    assert torch.allclose(losses[0], losses[1])
    assert len(gradients[0]) == len(gradients[1]) > 0
    for actual, expected in zip(gradients[0], gradients[1]):
        assert torch.allclose(actual, expected, atol=1e-6)
    # The offline weight would have given another loss.
    assert not torch.allclose(losses[0], losses[2])


# --- one buffer for the dataset and the online transitions --------------------

APPENDS = ["acfql", "fac", "fino", "floq", "fql", "value_flows"]


def test_default_replay_modes():
    modes = {
        name: entry.args_cls().online_replay_mode for name, entry in _registered("off2on").items()
    }
    assert {name for name, mode in modes.items() if mode == "append"} == set(APPENDS)
    assert {name for name, mode in modes.items() if mode == "empty"} == {"spot", "wsrl"}
    assert {name for name, mode in modes.items() if mode == "mixed"} == {"awac", "calql", "iql"}


def _default_mode_inputs(name, shared, tmp_path, with_dataset=True, **values):
    inputs = _inputs(name, shared, tmp_path, with_dataset=with_dataset, **values)
    assert "online_replay_mode" not in inputs
    return inputs


@pytest.mark.parametrize("name", APPENDS)
def test_online_batches_come_from_the_dataset_and_the_online_rows(
    name, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, result = _run(
        name, tmp_path / "run",
        _default_mode_inputs(name, shared_inputs, tmp_path, **OFFLINE_ONLINE), monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    buffer = agent.replay_buffer
    assert agent.offline_replay_buffer is None and len(buffer) == 48 + 16
    # The dataset's rows come first, untouched, then the online ones.
    fresh = _rebuild(name, tmp_path, shared_inputs, monkeypatch, "append")
    load_h5_dataset_to_replay_buffer(fresh.replay_buffer, str(shared_inputs.h5))
    assert torch.equal(buffer.actions[:24], fresh.replay_buffer.actions[:24])
    batch = [agent._sample_train_batch(64) for _ in range(8)]
    sampled = torch.cat([b.obs["state"] for b in batch])
    for rows in (buffer.obs["state"][:24], buffer.obs["state"][24:32]):
        rows = rows.reshape(-1, rows.shape[-1]).to(sampled.device)
        assert (sampled[:, None] == rows[None]).all(-1).any()
    losses = agent.train(2, compute_info=True)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values()), losses
    if name == "fac":
        assert "bc_flow_loss" in losses


def test_acfql_chunks_do_not_join_the_dataset_and_online_rows(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    """The dataset's last trajectory is complete for the windows even if the
    file flags neither its termination nor its truncation."""
    unflagged = tmp_path / "unflagged.h5"
    shutil.copy(shared_inputs.h5, unflagged)
    with h5py.File(unflagged, "r+") as dataset:
        for trajectory in dataset.values():
            trajectory["truncated"][...] = False
            trajectory["terminated"][...] = False
    for path in (shared_inputs.h5, unflagged):
        agent, result = _run(
            "acfql", tmp_path / path.stem,
            {**_default_mode_inputs("acfql", shared_inputs, tmp_path, **OFFLINE_ONLINE),
             "offline_dataset": str(path), "num_envs": "1"},
            monkeypatch,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        buffer = agent.replay_buffer
        ends = buffer.episode_ends[:64, 0].nonzero().flatten().tolist()
        assert {23, 47} <= set(ends)
        for row in (22, 23, 46, 47):
            rows, envs = torch.tensor([row]), torch.tensor([0])
            assert buffer._valid_nstep_batch(rows, envs).all()
            _, _, next_rows, _, valid = buffer._accumulate_chunk(rows, envs)
            last = 23 if row < 24 else 47
            assert valid[:, 0].tolist() == [True, row < last]
            assert next_rows.item() == last


@pytest.mark.parametrize("name", APPENDS)
def test_an_old_empty_run_does_not_become_append(
    name, tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent, checkpoint = _trained(name, "empty", shared_inputs, tmp_path, monkeypatch)
    resume = ["--load-checkpoint", str(checkpoint), "--load-replay-buffer"]

    def resumed_by_default(directory):
        inputs = _default_mode_inputs(
            name, shared_inputs, tmp_path, with_dataset=False, **CONTINUE_ONLY
        )
        return _run(name, tmp_path / directory, inputs, monkeypatch, *resume)

    for step in ("recorded", "run_config"):
        caplog.clear()
        if step == "run_config":
            _forget_stage(checkpoint, run_config=False)
        with caplog.at_level(logging.WARNING):
            resumed, result = resumed_by_default(step)
        assert result.returncode == 0, result.stderr[-1500:]
        assert _stage(resumed) == _stage(agent)
        assert "online_replay_mode=empty" in caplog.text

    contents = torch.load(checkpoint)
    del contents["metadata"]["run_config"]
    torch.save(contents, checkpoint)
    _, result = resumed_by_default("nothing")
    assert result.returncode == 1
    assert "pass --online-replay-mode" in result.stderr


def test_a_dataset_larger_than_the_buffer_keeps_its_newest_rows(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, result = _run(
        "fql", tmp_path / "run",
        {**_default_mode_inputs("fql", shared_inputs, tmp_path, num_offline_steps="2",
                                num_online_steps="0"), "buffer_size": "32", "num_envs": "1"},
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(agent.replay_buffer) == 32


# --- inputs a resume or a run without offline steps reads ----------------------


def test_an_unrecorded_stage_takes_the_warmup_its_metadata_records(
    tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    agent, checkpoint = _trained(
        "wsrl", "append", shared_inputs, tmp_path, monkeypatch, warmup_steps="100"
    )
    assert agent._active_initial_training_phase() is not None
    _forget_stage(checkpoint, run_config=True)
    inputs = _inputs("wsrl", shared_inputs, tmp_path, with_dataset=False, **MODES["append"],
                     **CONTINUE_ONLY)
    del inputs["warmup_steps"]
    flags = ["--load-checkpoint", str(checkpoint), "--load-replay-buffer"]
    with caplog.at_level(logging.WARNING):
        resumed, result = _run("wsrl", tmp_path / "default", inputs, monkeypatch, *flags)
    assert result.returncode == 0, result.stderr[-1500:]
    assert resumed.initial_training_phase == agent.initial_training_phase
    assert resumed._active_initial_training_phase() is not None
    assert "warmup_steps=100" in caplog.text
    _, result = _run("wsrl", tmp_path / "explicit", {**inputs, "warmup_steps": "10"},
                     monkeypatch, *flags)
    assert result.returncode == 1 and "--warmup-steps 10 (saved: 100)" in result.stderr


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("mode", ["append", "mixed"])
def test_a_run_without_offline_steps_keeps_its_dataset(
    name, mode, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, result = _run(
        name, tmp_path / "run",
        _inputs(name, shared_inputs, tmp_path, **MODES[mode],
                **EMPTY_MC.get((name, mode), {}), num_offline_steps="0", num_online_steps="16"),
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    if mode == "append":
        assert len(agent.replay_buffer) == 48 + 16
    else:
        assert (len(agent.replay_buffer), len(agent.offline_replay_buffer)) == (16, 48)
    # The setup the offline stage does on its dataset.
    if hasattr(agent, "fit_obs_normalizer"):
        means = [v for k, v in agent.policy.state_dict().items() if "obs_mean" in k]
        assert means and all((m != 0).any() for m in means)
    if hasattr(agent, "pretrain_vae"):
        assert agent._vae_pretrained
    if name == "fac":
        assert agent._logp_table is not None
        assert agent._phase_step == agent._bc_pretrain_steps_resolved > 0
    if isinstance(agent.replay_buffer, MCReplayBuffer) and mode == "append":
        assert agent.replay_buffer.sampleable_size >= 48


@pytest.mark.parametrize("name", ["iql", "fql"])
def test_without_a_dataset_a_run_without_offline_steps_is_unchanged(
    name, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, result = _run(
        name, tmp_path / "run",
        _inputs(name, shared_inputs, tmp_path, with_dataset=False, **MODES["append"],
                num_offline_steps="0", num_online_steps="16"),
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(agent.replay_buffer) == 16


def test_the_empty_mode_without_offline_steps_ignores_a_dataset(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, result = _run(
        "iql", tmp_path / "run",
        _inputs("iql", shared_inputs, tmp_path, **MODES["empty"], num_offline_steps="0",
                num_online_steps="0"),
        monkeypatch,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(agent.replay_buffer) == 0


@pytest.mark.parametrize("name", ["iql"])
def test_offline_steps_on_a_restored_buffer_need_no_dataset(
    name, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    _, checkpoint = _trained(name, "append", shared_inputs, tmp_path, monkeypatch)
    flags = ["--load-checkpoint", str(checkpoint.with_name("offline_final.pt"))]
    inputs = _inputs(
        name, shared_inputs, tmp_path, with_dataset=False, **MODES["append"],
        num_offline_steps="2", num_online_steps="0",
    )
    resumed, result = _run(name, tmp_path / "resumed", inputs, monkeypatch, *flags,
                           "--load-replay-buffer")
    assert result.returncode == 0, result.stderr[-1500:]
    assert len(resumed.replay_buffer) == 48 and resumed._online_start_step == 4
    _, result = _run(name, tmp_path / "unrestored", inputs, monkeypatch, *flags)
    assert result.returncode == 1 and "--offline_dataset is required" in result.stderr


def test_offline_steps_on_a_restored_empty_buffer_need_a_dataset(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent = _rebuild("iql", tmp_path, shared_inputs, monkeypatch, "append")
    path = agent.save(tmp_path / "empty.pt", include_replay_buffer=True)
    _, result = _run(
        "iql", tmp_path / "resumed",
        _inputs("iql", shared_inputs, tmp_path, with_dataset=False, **MODES["append"],
                warmup_steps="6", num_offline_steps="2", num_online_steps="0"),
        monkeypatch, "--load-checkpoint", str(path), "--load-replay-buffer",
    )
    assert result.returncode == 1 and "--offline_dataset is required" in result.stderr


@pytest.mark.parametrize("stage", ["offline_final.pt", "final.pt"])
@pytest.mark.parametrize("name,field", [("value_flows", "min_reward"), ("floq", "r_min")])
def test_a_resume_with_another_return_range_is_reported(
    name, field, stage, tmp_path, shared_inputs, monkeypatch, caplog  # noqa: F811
):
    _, checkpoint = _trained(
        name, "append", shared_inputs, tmp_path, monkeypatch, **{field: "-7.0"}
    )
    inputs = _inputs(name, shared_inputs, tmp_path, **MODES["append"], **CONTINUE_ONLY)
    flags = ["--load-checkpoint", str(checkpoint.with_name(stage)), "--load-replay-buffer"]
    for values, reported in (({}, True), ({field: "-7.0"}, False)):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _, result = _run(name, tmp_path / str(reported), {**inputs, **values}, monkeypatch,
                             *flags)
        assert result.returncode == 0, result.stderr[-1500:]
        flag = f"--{field.replace('_', '-')} -7.0"
        assert (f"{field}=-7.0 (now -1.0)" in caplog.text and flag in caplog.text) == reported
