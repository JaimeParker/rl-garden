"""Every registered algorithm that accepts ``--save-replay-buffer`` saves its
replay buffer and reloads it into an agent built by the real CLI builder."""
from __future__ import annotations

import dataclasses

import pytest
import torch

try:
    from tests._cli import run_entry
    from tests.test_entrypoint_builds import (
        EVAL_ONLY, EXEMPTIONS, PHASES, REGISTERED, _registered, build_argv,
        cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from tests.test_replay_buffer_state import assert_tree_equal, fill_randomly
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry
    from test_entrypoint_builds import (
        EVAL_ONLY, EXEMPTIONS, PHASES, REGISTERED, _registered, build_argv,
        cpu_only_and_restored_rng, inputs_for, shared_inputs,
    )
    from test_replay_buffer_state import assert_tree_equal, fill_randomly

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.mmap_multitask_episode_buffer import MmapMultitaskEpisodeBuffer
from rl_garden.training import inspection

__all__ = ["cpu_only_and_restored_rng", "shared_inputs"]  # fixtures

# (phase, algorithm) -> why --save-replay-buffer is refused.
REFUSALS = {("online", "ncrl"): "NCRL does not checkpoint its replay buffer"}

# On-policy algorithms, and offline ones that read their dataset in place. The
# flag comes from an args class they share and has no effect.
WITHOUT_REPLAY_BUFFER = {
    *(("online", name) for name in (
        "diffusion_cm_distill_online", "dppo", "flow_ppo", "policy_distillation",
        "ppo", "recurrent_ppo", "transformer_ppo",
    )),
    *(("offline", name) for name in (
        "a2a_bc", "consistency_distill_bc", "diffusion_bc", "hilp", "opal", "rssm_pretrain",
    )),
}

SHORT_WINDOWS = {
    "burn_in_len": "4", "memory_len": "4", "learning_len": "2", "forward_len": "2", "horizon": "2",
}


def _accepts_save_replay_buffer(phase: str, name: str) -> bool:
    fields = {field.name for field in dataclasses.fields(_registered(phase)[name].args_cls)}
    return "save_replay_buffer" in fields


CASES = sorted(
    case for case in REGISTERED - EXEMPTIONS.keys() if _accepts_save_replay_buffer(*case)
)


def _dry_run(phase, name, tmp_path, shared, monkeypatch, *flags: str):
    """The agent ``--dry-run`` builds and the CLI result."""
    built = []
    construct = inspection.construct_agent

    def capture(*args, **kwargs):
        built.append(construct(*args, **kwargs))
        return built[-1]

    inputs = {**SHORT_WINDOWS, "buffer_size": "64", **inputs_for(phase, name, shared, tmp_path)}
    if name == "supermimic":
        inputs.pop("horizon")
    argv = build_argv(
        phase, name, tmp_path / "runs", with_eval=(phase, name) in EVAL_ONLY, inputs=inputs
    )
    with monkeypatch.context() as patch:
        patch.setattr(inspection, "construct_agent", capture)
        result = run_entry(PHASES[phase][0], [*argv, "--dry-run", *flags])
    return (built[-1] if built else None), result


def _checkpointed_buffer(agent, name):
    """DAgger keeps what it aggregates in its demo buffer; its replay buffer is unused."""
    if name == "dagger":
        return agent.offline_replay_buffer
    return getattr(agent, "replay_buffer", None)


def _fill(buffer) -> None:
    if isinstance(buffer, MmapMultitaskEpisodeBuffer):
        for _ in range(2):
            length = buffer.horizon + 4
            buffer.load_episode(
                torch.randn(length, buffer.obs_dim),
                torch.randn(length, buffer.action_dim),
                torch.randn(length),
                0,
            )
    else:
        fill_randomly(buffer, 28)


def test_every_algorithm_accepting_the_flag_is_covered():
    assert REFUSALS.keys() <= set(CASES)
    assert len(CASES) >= 60


@pytest.mark.parametrize("phase,name", CASES, ids=[f"{p}-{n}" for p, n in CASES])
def test_replay_buffer_survives_save_build_load(
    phase, name, tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    if (phase, name) in REFUSALS:
        with pytest.raises(NotImplementedError, match=REFUSALS[(phase, name)]):
            _dry_run(
                phase, name, tmp_path, shared_inputs, monkeypatch, "--save-replay-buffer"
            )
        return
    agent, result = _dry_run(
        phase, name, tmp_path / "first", shared_inputs, monkeypatch, "--save-replay-buffer"
    )
    assert result.returncode == 0, result.stderr[-1500:]
    buffer = _checkpointed_buffer(agent, name)
    has_replay_buffer = isinstance(buffer, BaseReplayBuffer)
    assert has_replay_buffer == ((phase, name) not in WITHOUT_REPLAY_BUFFER)
    if not has_replay_buffer:
        return

    if len(buffer) == 0 and buffer.per_env_buffer_size > 1:
        _fill(buffer)
    path = tmp_path / "checkpoint_1.pt"
    agent.save(path, include_replay_buffer=True)

    fields = {field.name for field in dataclasses.fields(_registered(phase)[name].args_cls)}
    flags = ["--load-checkpoint", str(path)]
    if "load_replay_buffer" in fields:
        flags.append("--load-replay-buffer")
    loaded, result = _dry_run(
        phase, name, tmp_path / "second", shared_inputs, monkeypatch, *flags
    )
    assert result.returncode == 0, result.stderr[-1500:]
    if len(_checkpointed_buffer(loaded, name)) != len(buffer):
        # These entrypoints load after the dry run, or load weights only.
        loaded.load(path, load_replay_buffer=True)

    assert len(_checkpointed_buffer(loaded, name)) == len(buffer)
    if len(buffer) == 0:
        return
    samples = []
    for replay_buffer in (buffer, _checkpointed_buffer(loaded, name)):
        torch.manual_seed(0)
        samples.append([replay_buffer.sample(4) for _ in range(3)])
    assert_tree_equal(*samples)


def test_flash_sac_loads_its_replay_buffer_only_when_asked(
    tmp_path, shared_inputs, monkeypatch  # noqa: F811
):
    agent, _ = _dry_run(
        "online", "flash_sac", tmp_path / "first", shared_inputs, monkeypatch,
        "--checkpoint-dir", str(tmp_path / "checkpoints"),
    )
    assert str(agent.checkpoint_dir) == str(tmp_path / "checkpoints")
    _fill(agent.replay_buffer)
    path = tmp_path / "checkpoint_1.pt"
    agent.save(path, include_replay_buffer=True)
    for flags, size in ([], 0), (["--load-replay-buffer"], len(agent.replay_buffer)):
        loaded, result = _dry_run(
            "online", "flash_sac", tmp_path / "second", shared_inputs, monkeypatch,
            "--load-checkpoint", str(path), *flags,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert len(loaded.replay_buffer) == size
