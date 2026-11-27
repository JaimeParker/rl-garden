"""The checkpoint seams of ``BaseAlgorithm`` (``docs/guides/extending.md``,
section 4): ``_sidecar_buffers`` (buffers saved beside the replay file) and
``_on_checkpoint_read`` (a look at the checkpoint before anything is
restored), and the helpers in ``rl_garden.common.checkpoint`` they rest on."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from test_checkpoint import _add_state_transitions, _sac_kwargs, _state_env, _wsrl_kwargs
from test_dagger import _make_agent as _dagger
from test_fql_off2on import _agent as _off2on_fql
from test_fql_off2on import _fill_offline
from test_replay_buffer_state import assert_tree_equal

from rl_garden.algorithms import SAC, WSRL
from rl_garden.common.checkpoint import (
    UNRECORDED,
    SidecarBuffer,
    load_replay_buffer_file,
    record_file,
    recorded_file,
    referenced_file,
    sidecar_path_for_checkpoint,
)

# --- helpers -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "checkpoint,expected",
    [
        ("checkpoint_100.pt", "demo_buffer_100.pt"),
        ("final.pt", "demo_buffer_final.pt"),
        ("offline_final.pt", "demo_buffer_offline_final.pt"),
    ],
)
def test_sidecar_file_takes_the_checkpoints_suffix(checkpoint, expected, tmp_path):
    assert sidecar_path_for_checkpoint(tmp_path / checkpoint, "demo_buffer") == tmp_path / expected


def test_recorded_file_tells_a_missing_slot_from_none():
    extra = {"stage": {"path": None}, "named": "a.pt"}
    assert recorded_file(extra, ("named",)) == "a.pt"
    assert recorded_file(extra, ("stage", "path")) is None
    assert recorded_file(extra, ("stage", "other")) is UNRECORDED
    assert recorded_file(extra, ("absent", "path")) is UNRECORDED
    assert recorded_file(extra, ("named", "path")) is UNRECORDED


def test_record_file_needs_the_mapping_above_the_slot():
    extra: dict[str, Any] = {"stage": {}}
    record_file(extra, ("stage", "path"), "a.pt")
    record_file(extra, ("top",), "b.pt")
    assert extra == {"stage": {"path": "a.pt"}, "top": "b.pt"}
    with pytest.raises(KeyError, match="no mapping at absent"):
        record_file(extra, ("absent", "path"), "c.pt")


@pytest.mark.parametrize("exists", [True, False])
def test_referenced_file_resolves_a_name_none_and_a_missing_slot(exists, tmp_path):
    checkpoint = tmp_path / "checkpoint_1.pt"
    named, beside = tmp_path / "named.pt", tmp_path / "beside.pt"
    if exists:
        named.touch()
        beside.touch()

    def resolve(reference):
        return referenced_file(checkpoint, reference, unrecorded=beside, description="test buffer")

    if exists:
        assert resolve("named.pt") == named
    else:
        with pytest.raises(FileNotFoundError, match="references test buffer .*named.pt"):
            resolve("named.pt")
    assert resolve(None) is None
    assert resolve(UNRECORDED) == (beside if exists else None)
    assert referenced_file(checkpoint, UNRECORDED, unrecorded=None, description="x") is None


# --- _sidecar_buffers ------------------------------------------------------------------------


class TwoSidecarSAC(SAC):
    """Two extra buffers, one named at the top of ``extra``, one a level down."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first = self._build_replay_buffer()
        self.second = self._build_replay_buffer()
        self.restored: list[tuple[str, str, bool]] = []

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._extra_checkpoint_state(),
            "first_path": None,
            "group": {"second_path": None},
        }

    def _sidecar_buffers(self) -> tuple[SidecarBuffer, ...]:
        return (
            *super()._sidecar_buffers(),
            SidecarBuffer("first_buffer", ("first_path",), self.first, self._restore("first")),
            SidecarBuffer(
                "second_buffer", ("group", "second_path"), self.second, self._restore("second")
            ),
        )

    def _restore(self, name: str):
        def restore(path: Path, strict: bool) -> None:
            self.restored.append((name, path.name, strict))
            load_replay_buffer_file(path, getattr(self, name), strict=strict)

        return restore


def _two_sidecar_agent() -> TwoSidecarSAC:
    agent = TwoSidecarSAC(env=_state_env(), **_sac_kwargs())
    _add_state_transitions(agent)
    for buffer, steps in ((agent.first, 2), (agent.second, 3)):
        agent.replay_buffer, original = buffer, agent.replay_buffer
        _add_state_transitions(agent, steps)
        agent.replay_buffer = original
    return agent


def test_sidecar_buffers_are_written_named_and_restored(tmp_path):
    agent = _two_sidecar_agent()
    path = agent.save(tmp_path / "checkpoint_8.pt", include_replay_buffer=True)
    assert {p.name for p in tmp_path.iterdir()} == {
        "checkpoint_8.pt", "replay_buffer_8.pt", "first_buffer_8.pt", "second_buffer_8.pt",
    }
    extra = torch.load(path)["state"]["extra"]
    assert extra["first_path"] == "first_buffer_8.pt"
    assert extra["group"] == {"second_path": "second_buffer_8.pt"}

    loaded = TwoSidecarSAC(env=_state_env(), **_sac_kwargs())
    loaded.load(path, load_replay_buffer=False)
    assert loaded.restored == [] and len(loaded.first) == 0
    loaded.load(path, strict=False)
    assert loaded.restored == [
        ("first", "first_buffer_8.pt", False), ("second", "second_buffer_8.pt", False),
    ]
    saved_states = [buffer.state_dict() for buffer in (agent.first, agent.second, agent.replay_buffer)]
    for saved, buffer in zip(saved_states, (loaded.first, loaded.second, loaded.replay_buffer)):
        assert_tree_equal(buffer.state_dict(), saved)
    # The buffers hold different, non-constant data, so a restore into the wrong
    # one or a corrupted column cannot pass the comparison above.
    first, second, main = saved_states
    for state in saved_states:
        assert state["rewards"].unique().numel() > 1 and state["actions"].std() > 0
    assert not torch.equal(first["actions"][:2], second["actions"][:2])
    assert not torch.equal(first["actions"][:2], main["actions"][:2])
    assert (first["pos"], second["pos"], main["pos"]) == (2, 3, 4)


def test_a_save_without_replay_writes_and_names_no_sidecar(tmp_path):
    agent = _two_sidecar_agent()
    agent.second = None
    path = agent.save(tmp_path / "with_replay.pt", include_replay_buffer=True)
    extra = torch.load(path)["state"]["extra"]
    assert extra["first_path"] == "first_buffer_with_replay.pt"
    assert extra["group"] == {"second_path": None}
    assert not (tmp_path / "second_buffer_with_replay.pt").exists()

    path = _two_sidecar_agent().save(tmp_path / "checkpoint_8.pt")
    extra = torch.load(path)["state"]["extra"]
    assert extra["first_path"] is None and extra["group"] == {"second_path": None}
    assert {p.name for p in tmp_path.iterdir()} == {
        "with_replay.pt", "replay_buffer_with_replay.pt", "first_buffer_with_replay.pt",
        "checkpoint_8.pt",
    }
    loaded = TwoSidecarSAC(env=_state_env(), **_sac_kwargs())
    loaded.load(path)
    assert loaded.restored == []


# --- _on_checkpoint_read ------------------------------------------------------------------


class ReadRecordingSAC(SAC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.events: list[tuple[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _on_checkpoint_read(self, checkpoint, *, replay_path) -> None:
        super()._on_checkpoint_read(checkpoint, replay_path=replay_path)
        self.events.append(("read", replay_path))
        checkpoint["state"]["extra"]["filled_in"] = True

    def _load_extra_checkpoint_state(self, state) -> None:
        super()._load_extra_checkpoint_state(state)
        self.events.append(("extra", state.get("filled_in")))


def test_checkpoint_is_shown_before_any_state_is_restored(tmp_path):
    agent = ReadRecordingSAC(env=_state_env(), **_sac_kwargs())
    _add_state_transitions(agent)
    path = agent.save(tmp_path / "checkpoint_8.pt", include_replay_buffer=True)

    for load_replay_buffer, replay_path in ((False, None), (True, tmp_path / "replay_buffer_8.pt")):
        loaded = ReadRecordingSAC(env=_state_env(), **_sac_kwargs())
        loaded.load(path, load_replay_buffer=load_replay_buffer)
        assert loaded.events == [("read", replay_path), ("extra", True)]

    other = ReadRecordingSAC(env=_state_env(), **{**_sac_kwargs(), "net_arch": [8]})
    with pytest.raises(RuntimeError):
        other.load(path)
    assert other.events == [("read", tmp_path / "replay_buffer_8.pt")]


def test_an_invalid_checkpoint_is_rejected_before_it_is_shown(tmp_path):
    path = ReadRecordingSAC(env=_state_env(), **_sac_kwargs()).save(tmp_path / "a.pt")
    contents = torch.load(path)
    contents["metadata"]["algorithm_class"] = "PPO"
    torch.save(contents, path)
    loaded = ReadRecordingSAC(env=_state_env(), **_sac_kwargs())
    with pytest.raises(ValueError, match="algorithm mismatch"):
        loaded.load(path)
    assert loaded.events == []


# --- one read of the checkpoint file per load() ----------------------------------------------


def _online_wsrl() -> WSRL:
    agent = WSRL(env=_state_env(), **_wsrl_kwargs())
    _add_state_transitions(agent)
    agent.switch_to_online_mode("mixed", 0.5)
    _add_state_transitions(agent)
    return agent


def _online_fql():
    agent = _off2on_fql()
    _fill_offline(agent)
    agent.switch_to_online_mode("mixed", 0.5)
    return agent


def _dagger_with_demo_rows():
    agent = _dagger()
    agent.collect_round(num_steps=4, beta=1.0)
    return agent


@pytest.mark.parametrize("make", [_online_wsrl, _online_fql, _dagger_with_demo_rows])
def test_load_reads_the_checkpoint_file_once(make, tmp_path, monkeypatch):
    path = make().save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
    reads: list[Path] = []
    original = torch.load

    def counting_load(file, *args, **kwargs):
        reads.append(Path(file))
        return original(file, *args, **kwargs)

    agent = make()
    monkeypatch.setattr(torch, "load", counting_load)
    agent.load(path)
    assert reads.count(path) == 1
    sidecars = {p for p in reads if p != path}
    assert sidecars == {p for p in tmp_path.iterdir() if p != path}
