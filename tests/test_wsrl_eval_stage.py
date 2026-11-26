"""Evaluation follows the stage a checkpoint was saved in, without ``--phase``
or ``--algorithm``: a WSRL checkpoint of the offline stage is a Cal-QL policy,
one of the online stage a SAC policy."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import test_checkpoint_eval_is_self_contained as self_contained
import test_entrypoint_builds as builds
import torch

from rl_garden.algorithms import CQL, Off2OnIQL, WSRL, CalQL
from rl_garden.buffers import ReplayBuffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer
from rl_garden.training import evaluation

shared_inputs = builds.shared_inputs
cpu_only_and_restored_rng = builds.cpu_only_and_restored_rng

OFFLINE, ONLINE = "offline (Cal-QL)", "online (SAC)"
OBSERVATION = {"state": torch.tensor([[0.3, -0.8, 0.5], [-1.0, 0.1, -2.0]])}


@pytest.fixture
def saved_actions(monkeypatch):
    """Checkpoint file name -> the saving agent's deterministic actions."""
    actions = {}
    for cls in (WSRL, CQL, CalQL, Off2OnIQL):
        save = cls.save

        def recording(agent, path, *args, save=save, **kwargs):
            actions[Path(path).name] = _actions(agent)
            return save(agent, path, *args, **kwargs)

        monkeypatch.setattr(cls, "save", recording)
    return actions


def _actions(agent) -> torch.Tensor:
    agent.policy.eval()
    actions = agent._eval_action(OBSERVATION).clone()
    agent.policy.train()
    return actions


def _train(phase, name, shared, tmp_path, **values) -> Path:
    inputs = {
        **self_contained.SMALL_TRAINING, "offline_dataset": str(shared.h5), "action_low": "-2",
        "action_high": "2", "learning_starts": "0", "warmup_steps": "0", "checkpoint_freq": "8",
        **values,
    }
    argv = builds.build_argv(phase, name, tmp_path / "runs", inputs=inputs)
    result = builds.run_entry(builds.PHASES[phase][0], argv)
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = (tmp_path / "runs").glob("*/checkpoints")
    return directory


def _with_source(checkpoint: Path, source: str, tmp_path: Path) -> tuple[Path, dict]:
    """The checkpoint with its run configuration embedded, only beside it, or
    nowhere; and the eval flags that then says what is missing."""
    run = tmp_path / source / "run"
    copied = run / "checkpoints" / checkpoint.name
    copied.parent.mkdir(parents=True)
    shutil.copy(checkpoint, copied)
    if source == "run_config":
        return copied, {}
    contents = torch.load(copied)
    del contents["metadata"]["run_config"]
    torch.save(contents, copied)
    if source == "config_json":
        shutil.copy(checkpoint.parent.parent / "config.json", run)
        return copied, {}
    return copied, {"env_backend": "gymnasium", "env_id": self_contained.STATE_ENV_ID}


SOURCES = ("run_config", "config_json", "neither")


def _evaluate(checkpoint, source, tmp_path, monkeypatch, phase="auto", algorithm="auto"):
    copied, flags = _with_source(checkpoint, source, tmp_path)
    return self_contained._evaluate(copied, phase, algorithm, monkeypatch, **flags)


@pytest.mark.parametrize("source", SOURCES)
def test_pretrained_wsrl_is_a_calql_policy(
    source, tmp_path, shared_inputs, saved_actions, monkeypatch
):
    directory = _train("offline", "wsrl", shared_inputs, tmp_path)
    checkpoint = directory / "wsrl_offline_pretrained.pt"
    result, agent = _evaluate(checkpoint, source, tmp_path, monkeypatch)
    assert (result["phase"], result["algorithm"]) == ("offline", "wsrl")
    assert result["training_stage"] == OFFLINE
    _assert_offline(agent)
    assert torch.equal(_actions(agent), saved_actions[checkpoint.name])


def _assert_offline(agent) -> None:
    assert agent._online_start_step is None
    assert agent.use_cql_loss and agent.use_calql and agent.cql_alpha == 5.0
    assert agent.offline_replay_buffer is None
    assert type(agent.replay_buffer) is MCReplayBuffer


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("mode", ["empty", "append", "mixed"])
def test_off2on_wsrl_changes_from_calql_to_sac(
    mode, source, tmp_path, shared_inputs, saved_actions, monkeypatch
):
    directory = _train(
        "off2on", "wsrl", shared_inputs, tmp_path, online_replay_mode=mode,
        offline_data_ratio="0.5", num_offline_steps="8", num_online_steps="16",
        actor_hidden_layers="2", critic_hidden_layers="1",
    )
    stages = {
        # Written by the offline loop of the off2on run.
        "checkpoint_8.pt": OFFLINE,
        "offline_final.pt": OFFLINE,
        "checkpoint_16.pt": ONLINE,
        "final.pt": ONLINE,
    }
    for name, stage in stages.items():
        result, agent = _evaluate(directory / name, source, tmp_path / name, monkeypatch)
        assert result["training_stage"] == stage, name
        if source == "run_config":
            assert result["phase"] == "off2on"
        else:
            assert result["phase"] == ("offline" if stage is OFFLINE else "off2on")
        if stage is OFFLINE:
            _assert_offline(agent)
        else:
            assert agent._online_start_step == 8
            assert agent.use_cql_loss is False and agent.cql_alpha == 0.0
            assert agent.offline_replay_buffer is None
            expected = ReplayBuffer if mode == "empty" else MCReplayBuffer
            assert type(agent.replay_buffer) is expected
        assert torch.equal(_actions(agent), saved_actions[name]), name


@pytest.mark.parametrize("name", ["cql", "calql"])
@pytest.mark.parametrize("source", SOURCES)
def test_cql_family_checkpoint_through_wsrl(
    name, source, tmp_path, shared_inputs, saved_actions, monkeypatch
):
    directory = _train("offline", name, shared_inputs, tmp_path)
    checkpoint = directory / f"{name}_offline_pretrained.pt"
    result, agent = _evaluate(checkpoint, source, tmp_path, monkeypatch, algorithm="wsrl")
    assert isinstance(agent, WSRL)
    assert (result["phase"], result["algorithm"]) == ("offline", "wsrl")
    assert result["training_stage"] == OFFLINE
    assert torch.equal(_actions(agent), saved_actions[checkpoint.name])

    result, _ = _evaluate(checkpoint, source, tmp_path / "own", monkeypatch)
    assert result["algorithm"] == name
    assert "training_stage" not in result


def test_online_objective_follows_online_use_cql_loss(
    tmp_path, shared_inputs, saved_actions, monkeypatch
):
    directory = _train(
        "off2on", "wsrl", shared_inputs, tmp_path, online_replay_mode="append",
        online_use_cql_loss=True, online_cql_alpha="0.5",
    )
    result, agent = _evaluate(directory / "final.pt", "run_config", tmp_path, monkeypatch)
    assert result["training_stage"] == "online (Cal-QL)"
    assert agent.use_cql_loss and agent.cql_alpha == 0.5
    assert torch.equal(_actions(agent), saved_actions["final.pt"])


@pytest.mark.parametrize("source", SOURCES)
def test_twice_resumed_checkpoint(
    source, tmp_path, shared_inputs, saved_actions, monkeypatch
):
    flags = {"online_replay_mode": "mixed", "offline_data_ratio": "0.5"}
    directory = _train("off2on", "wsrl", shared_inputs, tmp_path / "0", **flags)
    for run in (1, 2):
        directory = _train(
            "off2on", "wsrl", shared_inputs, tmp_path / str(run), **flags,
            num_offline_steps="0", load_checkpoint=str(directory / "final.pt"),
        )
    result, agent = _evaluate(directory / "final.pt", source, tmp_path, monkeypatch)
    assert (result["phase"], result["training_stage"]) == ("off2on", ONLINE)
    assert agent._online_start_step == 2 and agent._global_step == 2 + 3 * 16
    assert agent.use_cql_loss is False
    assert torch.equal(_actions(agent), saved_actions["final.pt"])


def test_a_stage_the_checkpoint_does_not_record_is_not_reported_as_offline(
    tmp_path, shared_inputs, monkeypatch
):
    directory = _train("off2on", "wsrl", shared_inputs, tmp_path)
    checkpoint = directory / "final.pt"
    contents = torch.load(checkpoint)
    del contents["state"]["extra"]["online_start_step"]
    torch.save(contents, checkpoint)
    result, _ = _evaluate(checkpoint, "run_config", tmp_path, monkeypatch)
    assert result["training_stage"] == "unknown"


@pytest.mark.parametrize(
    "net_arch,expressible",
    [([12, 12], True), ({"pi": [12], "qf": [12, 12]}, True), ({"pi": [12, 8], "qf": [12]}, False)],
)
def test_recorded_net_arch(net_arch, expressible, tmp_path, shared_inputs, monkeypatch):
    directory = _train("off2on", "wsrl", shared_inputs, tmp_path)
    _, built = _evaluate(directory / "final.pt", "run_config", tmp_path / "template", monkeypatch)
    agent = WSRL(env=built.env, net_arch=net_arch, buffer_device="cpu", device="cpu")
    agent.run_config = {}
    path = agent.save(tmp_path / "run" / "checkpoints" / "final.pt")
    actions = _actions(agent)
    if not expressible:
        with pytest.raises(SystemExit, match="cannot express"):
            _evaluate(path, "neither", tmp_path, monkeypatch, phase="off2on")
        return
    _, evaluated = _evaluate(path, "neither", tmp_path, monkeypatch, phase="off2on")
    assert torch.equal(_actions(evaluated), actions)


def test_other_off2on_algorithms_report_the_bare_stage(
    tmp_path, shared_inputs, saved_actions, monkeypatch, capsys
):
    directory = _train("off2on", "iql", shared_inputs, tmp_path, online_replay_mode="append")
    for name, stage in (("offline_final.pt", "offline"), ("final.pt", "online")):
        result, agent = _evaluate(directory / name, "neither", tmp_path / name, monkeypatch)
        assert result["training_stage"] == stage
        assert result["phase"] == "off2on"
        assert torch.equal(_actions(agent), saved_actions[name])


def test_stage_is_printed_and_written_to_json(tmp_path, shared_inputs, monkeypatch, capsys):
    directory = _train("off2on", "wsrl", shared_inputs, tmp_path)
    output = tmp_path / "eval.json"
    monkeypatch.setattr(
        "sys.argv",
        ["rl-garden-eval", "--checkpoint-path", str(directory / "final.pt"), "--device", "cpu",
         "--num-eval-envs", "1", "--num-eval-episodes", "1", "--output-json", str(output)],
    )
    evaluation.main()
    assert f"training_stage: {ONLINE}" in capsys.readouterr().out
    assert json.loads(output.read_text())["training_stage"] == ONLINE


# --- resolution --------------------------------------------------------------


def _checkpoint(algorithm_class, selection=None, **extra):
    run_config = {} if selection is None else {
        "selection": {"training_phase": selection[0], "algorithm": selection[1]}
    }
    return {
        "metadata": {"algorithm_class": algorithm_class, "run_config": run_config},
        "state": {"extra": extra},
    }


@pytest.mark.parametrize(
    "requested,checkpoint,expected",
    [
        # The embedded selection decides, also for a class no entry declares.
        (("auto", "auto"), _checkpoint("SACDDP", ("online", "sac")), ("online", "sac")),
        (("auto", "auto"), _checkpoint("WSRL", ("off2on", "wsrl"), online_start_step=None),
         ("off2on", "wsrl")),
        (("auto", "wsrl"), _checkpoint("WSRL", ("off2on", "wsrl"), online_start_step=None),
         ("off2on", "wsrl")),
        # Explicit flags win over it.
        (("offline", "wsrl"), _checkpoint("WSRL", ("off2on", "wsrl")), ("offline", "wsrl")),
        (("offline", "auto"), _checkpoint("WSRL", ("off2on", "wsrl")), ("offline", "wsrl")),
        (("auto", "ppo"), _checkpoint("PPO", ("online", "recurrent_ppo")), ("online", "ppo")),
        # One naming no registered entry is ignored.
        (("auto", "auto"), _checkpoint("PPO", ("online", "gone")), ("online", "ppo")),
        (("auto", "auto"), _checkpoint("PPO", ("sideways", "ppo")), ("online", "ppo")),
        (("auto", "auto"), _checkpoint("WSRL", ("off2on", "gone"), online_start_step=None),
         ("offline", "wsrl")),
        # The recorded stage.
        (("auto", "auto"), _checkpoint("WSRL", online_start_step=None), ("offline", "wsrl")),
        (("auto", "auto"), _checkpoint("WSRL", online_start_step=0), ("off2on", "wsrl")),
        (("auto", "wsrl"), _checkpoint("WSRL", online_start_step=5), ("off2on", "wsrl")),
        (("off2on", "auto"), _checkpoint("WSRL", online_start_step=None), ("off2on", "wsrl")),
        (("auto", "wsrl"), _checkpoint("CalQL"), ("offline", "wsrl")),
        # An entry of the checkpoint's own class is preferred to the stage.
        (("auto", "iql"), _checkpoint("Off2OnIQL", online_start_step=None), ("off2on", "iql")),
        (("auto", "iql"), _checkpoint("IQL"), ("offline", "iql")),
    ],
)
def test_resolution(requested, checkpoint, expected):
    assert evaluation._resolve_phase_and_algorithm(*requested, checkpoint) == expected


def test_wsrl_without_a_recorded_stage_stays_ambiguous():
    with pytest.raises(SystemExit, match="ambiguous"):
        evaluation._resolve_phase_and_algorithm("auto", "auto", _checkpoint("WSRL"))
