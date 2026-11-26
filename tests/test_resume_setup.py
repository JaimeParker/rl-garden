"""One-time setup on a dataset (observation statistics, VAE pretraining) is not
repeated on resume, while the dataset itself is loaded wherever a run keeps it."""
from __future__ import annotations

import h5py
import pytest
import torch

try:
    from tests import test_entrypoint_builds as builds
except ImportError:
    # Some environments install a top-level tests package that shadows this one.
    import test_entrypoint_builds as builds

from rl_garden.algorithms.plas import PLASCore
from rl_garden.algorithms.spot import SPOTCore
from rl_garden.training import inspection

shared_inputs = builds.shared_inputs
cpu_only_and_restored_rng = builds.cpu_only_and_restored_rng

TRAINING = {
    "env_id": builds.LOCOMOTION_NAMED_PENDULUM_ID, "eval_freq": "0", "action_low": "-2",
    "action_high": "2", "training_freq": "4", "utd": "0.5", "learning_starts": "0",
    "warmup_steps": "0", "vae_iterations": "2", "num_offline_steps": "2", "num_online_steps": "16",
}


@pytest.fixture(scope="module")
def other_h5(tmp_path_factory):
    """A dataset with other observation statistics."""
    path = builds.write_dataset(tmp_path_factory.mktemp("other") / "other.h5")
    with h5py.File(path, "r+") as dataset:
        for trajectory in dataset.values():
            trajectory["obs"][...] = trajectory["obs"][...] * 5 + 3
    return path


@pytest.fixture
def vae_trainings(monkeypatch):
    """For each ``pretrain_vae`` call: the size of the replay buffer it saw, and
    whether the VAE's weights changed and how many optimizer steps it took."""
    calls = []
    for core in (SPOTCore, PLASCore):
        pretrain_vae = core.pretrain_vae

        def recorded(agent, pretrain_vae=pretrain_vae):
            size = len(agent.replay_buffer)
            before = [parameter.clone() for parameter in agent.policy.vae.parameters()]
            step, steps = agent.vae_optimizer.step, []
            agent.vae_optimizer.step = lambda: (steps.append(None), step())
            try:
                result = pretrain_vae(agent)
            finally:
                agent.vae_optimizer.step = step
            changed = any(
                not torch.equal(old, new)
                for old, new in zip(before, agent.policy.vae.parameters())
            )
            assert changed == bool(steps)
            calls.append((size, len(steps)))
            return result

        monkeypatch.setattr(core, "pretrain_vae", recorded)
    return calls


def _run(phase, name, log_dir, dataset, monkeypatch, *flags, **values):
    """``(agent, checkpoint directory)`` of a real run."""
    built = []
    construct = inspection.construct_agent

    def capture(*args, **kwargs):
        built.append(construct(*args, **kwargs))
        return built[-1]

    argv = builds.build_argv(
        phase, name, log_dir,
        inputs={**TRAINING, **({} if dataset is None else {"offline_dataset": str(dataset)}),
                **values},
    )
    with monkeypatch.context() as patch:
        patch.setattr(inspection, "construct_agent", capture)
        result = builds.run_entry(builds.PHASES[phase][0], [*argv, *flags])
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = log_dir.glob("*/checkpoints")
    return built[-1], directory


def _statistics(policy_state):
    found = {k: v for k, v in policy_state.items() if "obs_mean" in k or "obs_std" in k}
    assert found, list(policy_state)
    return found


def _assert_same_statistics(agent, checkpoint):
    saved = _statistics(torch.load(checkpoint)["state"]["policy"])
    assert any((value != 0).any() for key, value in saved.items() if "obs_mean" in key)
    for key, value in _statistics(agent.policy.state_dict()).items():
        assert torch.equal(value, saved[key]), key


@pytest.mark.parametrize("more_offline_steps", ["0", "2"])
@pytest.mark.parametrize("name", ["awac", "spot"])
def test_off2on_resume_of_an_offline_stage_keeps_the_statistics(
    name, more_offline_steps, tmp_path, shared_inputs, other_h5, vae_trainings, monkeypatch
):
    _, directory = _run(
        "off2on", name, tmp_path / "first", shared_inputs.h5, monkeypatch,
        online_replay_mode="append",
    )
    del vae_trainings[:]
    checkpoint = directory / "offline_final.pt"
    agent, _ = _run(
        "off2on", name, tmp_path / "second", other_h5, monkeypatch,
        "--load-checkpoint", str(checkpoint),
        online_replay_mode="append", num_offline_steps=more_offline_steps,
    )
    _assert_same_statistics(agent, checkpoint)
    assert len(agent.replay_buffer) == 48 + 16
    assert vae_trainings == ([(48, 0)] if name == "spot" else [])


@pytest.mark.parametrize("name", ["awac", "spot"])
@pytest.mark.parametrize("mode", ["append", "mixed"])
def test_off2on_resume_of_an_online_stage_keeps_the_statistics(
    name, mode, tmp_path, shared_inputs, other_h5, vae_trainings, monkeypatch
):
    modes = {"online_replay_mode": mode, "offline_data_ratio": "0.5" if mode == "mixed" else "0.0"}
    _, directory = _run("off2on", name, tmp_path / "first", shared_inputs.h5, monkeypatch, **modes)
    del vae_trainings[:]
    checkpoint = directory / "final.pt"
    agent, _ = _run(
        "off2on", name, tmp_path / "second", other_h5, monkeypatch,
        "--load-checkpoint", str(checkpoint), num_offline_steps="0", **modes,
    )
    _assert_same_statistics(agent, checkpoint)
    assert sum(steps for _, steps in vae_trainings) == 0
    if name == "spot":
        for key, value in torch.load(checkpoint)["state"]["policy"].items():
            if key.startswith("vae."):
                assert torch.equal(agent.policy.state_dict()[key], value), key


@pytest.mark.parametrize("name", ["awac", "td3_bc", "bcq", "plas", "spot"])
def test_offline_resume_repeats_no_setup(
    name, tmp_path, shared_inputs, other_h5, vae_trainings, monkeypatch
):
    trains_vae = name in {"plas", "spot"}
    _, directory = _run("offline", name, tmp_path / "first", shared_inputs.h5, monkeypatch)
    assert vae_trainings == ([(48, 2)] if trains_vae else [])
    (checkpoint,) = directory.glob("*.pt")
    vae = {
        key: value for key, value in torch.load(checkpoint)["state"]["policy"].items()
        if trains_vae and key.startswith("vae.")
    }
    assert bool(vae) == trains_vae

    del vae_trainings[:]
    agent, _ = _run(
        "offline", name, tmp_path / "second", other_h5, monkeypatch,
        "--load-checkpoint", str(checkpoint),
    )
    _assert_same_statistics(agent, checkpoint)
    assert vae_trainings == ([(48, 0)] if trains_vae else [])
    for key, value in vae.items():
        assert torch.equal(agent.policy.state_dict()[key], value), key


@pytest.mark.parametrize("phase,name", [("offline", "plas"), ("offline", "spot"), ("off2on", "spot")])
def test_a_checkpoint_without_a_pretrained_vae_gets_one(
    phase, name, tmp_path, shared_inputs, vae_trainings, monkeypatch
):
    _, directory = _run(phase, name, tmp_path / "first", shared_inputs.h5, monkeypatch)
    checkpoint = next(path for path in directory.glob("*.pt") if path.stem != "final")
    contents = torch.load(checkpoint)
    extra = contents["state"]["extra"]
    training = contents["state"]["training_state"]
    (extra if "vae_pretrained" in extra else training)["vae_pretrained"] = False
    torch.save(contents, checkpoint)
    del vae_trainings[:]
    agent, _ = _run(
        phase, name, tmp_path / "second", shared_inputs.h5, monkeypatch,
        "--load-checkpoint", str(checkpoint),
    )
    assert vae_trainings == [(48, 2)]
    assert agent._vae_pretrained
    assert not any(p.requires_grad for p in agent.policy.vae.parameters())
    if phase == "offline":
        _assert_same_statistics(agent, checkpoint)


@pytest.mark.parametrize("mode", ["append", "mixed"])
@pytest.mark.parametrize("restore_replay", [False, True])
def test_spot_online_stage_without_a_vae_pretrains_it_on_the_offline_data(
    mode, restore_replay, tmp_path, shared_inputs, vae_trainings, monkeypatch
):
    modes = {
        "online_replay_mode": mode, "offline_data_ratio": "auto" if mode == "mixed" else "0.0",
        "num_offline_steps": "0",
    }
    # Online from the start without a dataset: no VAE.
    trained, directory = _run(
        "off2on", "spot", tmp_path / "first", None, monkeypatch, **modes,
        save_replay_buffer=restore_replay,
    )
    assert vae_trainings == [] and not trained._vae_pretrained
    agent, _ = _run(
        "off2on", "spot", tmp_path / "second", shared_inputs.h5, monkeypatch,
        "--load-checkpoint", str(directory / "final.pt"), "--load-replay-buffer", **modes,
    )
    if mode == "mixed" and restore_replay:
        # The restored offline buffer is empty: there is nothing to train on.
        assert vae_trainings == []
        return
    assert vae_trainings == [(16 if restore_replay else 48, 2)]
    assert agent._vae_pretrained
    if mode == "mixed":
        assert (len(agent.replay_buffer), len(agent.offline_replay_buffer)) == (16, 48)


def test_rlpd_resume_has_both_buffers_once(tmp_path, shared_inputs, monkeypatch):
    values = {
        "offline_buffer_size": "96", "offline_data_ratio": "0.25", "total_timesteps": "16",
        "save_replay_buffer": True,
    }
    trained, directory = _run(
        "online", "rlpd", tmp_path / "first", shared_inputs.h5, monkeypatch, **values
    )
    resumed, _ = _run(
        "online", "rlpd", tmp_path / "second", shared_inputs.h5, monkeypatch,
        "--load-checkpoint", str(directory / "final.pt"), "--load-replay-buffer", **values,
    )
    for agent in (trained, resumed):
        assert len(agent.replay_buffer) == 16
        assert len(agent.offline_replay_buffer) == 48
        assert agent.offline_data_ratio == 0.25
    for name in ("obs", "rewards"):
        for buffer in ("replay_buffer", "offline_replay_buffer"):
            first, second = (getattr(getattr(a, buffer), name) for a in (trained, resumed))
            if name == "obs":
                first, second = first["state"], second["state"]
            assert torch.equal(first[: len(first)], second[: len(second)])


@pytest.mark.parametrize("name", ["iql", "calql", "cql", "fql", "spot", "wsrl"])
def test_offline_run_with_a_restored_buffer_holds_each_row_once(
    name, tmp_path, shared_inputs, monkeypatch
):
    trained, directory = _run(
        "offline", name, tmp_path / "first", shared_inputs.h5, monkeypatch,
        save_replay_buffer=True,
    )
    (checkpoint,) = (p for p in directory.glob("*.pt") if "replay_buffer" not in p.name)
    agent, _ = _run(
        "offline", name, tmp_path / "second", shared_inputs.h5, monkeypatch,
        "--load-checkpoint", str(checkpoint), "--load-replay-buffer",
    )
    assert len(agent.replay_buffer) == len(trained.replay_buffer) == 48
    assert torch.equal(agent.replay_buffer.actions[:24], trained.replay_buffer.actions[:24])
