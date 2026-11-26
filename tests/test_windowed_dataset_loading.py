"""Datasets loaded into buffers that sample windows of consecutive rows
(n-step returns, action chunks): no window leaves its trajectory."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

try:
    from tests._cli import run_entry
    from tests.test_entrypoint_builds import (
        PHASES,
        build_argv,
        cpu_only_and_restored_rng,
        write_dataset,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry
    from test_entrypoint_builds import PHASES, build_argv, cpu_only_and_restored_rng, write_dataset

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.algorithms.offline import OfflineRLAlgorithm
from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.buffers.dataset_common import add_flat_transitions
from rl_garden.buffers.h5_dataset import load_h5_dataset_to_replay_buffer
from rl_garden.buffers.mc_buffer import MCReplayBuffer
from rl_garden.buffers.nstep_buffer import NStepReplayBuffer
from rl_garden.buffers.rebrac_replay_buffer import ReBRACReplayBuffer
from rl_garden.buffers.sarsa_buffer import SarsaMCReplayBuffer

__all__ = ["cpu_only_and_restored_rng"]  # fixture

H = 3
GAMMA = 0.9
LENGTHS = (7, 2, 9, 5, 8)
OBS_SPACE = spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (2,), np.float32)})
ACT_SPACE = spaces.Box(-1.0, 1.0, (2,), np.float32)


def _buffer(kind: str, num_envs: int):
    size = {"num_envs": num_envs, "buffer_size": num_envs * 40}
    common = {"gamma": GAMMA, "storage_device": "cpu", "sample_device": "cpu", **size}
    if kind == "nstep":
        return NStepReplayBuffer(OBS_SPACE, ACT_SPACE, nstep=H, **common)
    return ChunkedReplayBuffer(OBS_SPACE, ACT_SPACE, horizon_length=H, **common)


def _is_terminal(ending: str, traj: int) -> bool:
    return ending == "terminated" and traj % 2 == 0


def _write_h5(path, ending: str):
    """Row ``t`` of trajectory ``i`` holds ``(i, t)``; ``ending`` says which fields mark its end."""
    with h5py.File(path, "w") as f:
        for i, length in enumerate(LENGTHS):
            group = f.create_group(f"traj_{i}")
            steps = np.arange(length + 1, dtype=np.float32)
            group["obs"] = np.column_stack((np.full(length + 1, i, np.float32), steps))
            group["actions"] = np.column_stack((np.full(length, i, np.float32), steps[:-1]))
            group["rewards"] = 10.0 * i + steps[:-1] + 1
            last = np.arange(length) == length - 1
            group["terminated"] = last & _is_terminal(ending, i)
            if ending != "unmarked":
                group["truncated"] = last & (not _is_terminal(ending, i))
    return path


def _candidates(buffer):
    t0 = torch.arange(buffer.size).repeat_interleave(buffer.num_envs)
    env = torch.arange(buffer.num_envs).repeat(buffer.size)
    valid = buffer._valid_nstep_batch(t0, env)
    buffer._sample_valid_indices = lambda n, upper: (t0[valid], env[valid])
    return t0[valid], env[valid], buffer.sample(int(valid.sum()))


@pytest.mark.parametrize("ending", ["truncated", "terminated", "unmarked"])
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("kind", ["nstep", "chunked"])
def test_windows_stay_inside_their_trajectory(kind, num_envs, ending, tmp_path):
    buffer = _buffer(kind, num_envs)
    loaded = load_h5_dataset_to_replay_buffer(buffer, _write_h5(tmp_path / "d.h5", ending))
    assert loaded == sum(LENGTHS) // num_envs * num_envs
    t0, _, sample = _candidates(buffer)
    assert len(t0) == loaded  # every stored transition starts a window
    per_env = loaded // num_envs
    full_windows = 0
    for i in range(len(t0)):
        traj, t = (int(v) for v in sample.obs["state"][i])
        stretch_left = per_env - int(t0[i])  # a column holds one contiguous stretch
        k = min(H, LENGTHS[traj] - t, stretch_left)
        full_windows += k == H
        assert sample.next_obs["state"][i].tolist() == [traj, t + k]
        expected = sum(GAMMA**j * (10.0 * traj + t + j + 1) for j in range(k))
        assert float(sample.rewards[i]) == pytest.approx(expected, rel=1e-5)
        terminal = _is_terminal(ending, traj) and t + k == LENGTHS[traj]
        assert float(sample.discounts[i]) == pytest.approx(0.0 if terminal else GAMMA**k)
        assert bool(sample.dones[i]) == terminal
        if kind == "chunked":
            assert sample.valid[i].tolist() == [j < k for j in range(H)]
            for j in range(k):
                assert sample.actions[i, j].tolist() == [traj, t + j]
    assert full_windows > loaded // 2


@pytest.mark.parametrize("kind", ["nstep", "chunked"])
def test_loading_without_episode_ends_is_refused(kind):
    rows = {"state": torch.zeros(6, 2)}
    with pytest.raises(ValueError, match="where episodes end"):
        add_flat_transitions(
            _buffer(kind, 1), rows, rows, torch.zeros(6, 2), torch.zeros(6), torch.zeros(6)
        )


CASES = {
    "rlpd": ("online", "rlpd", {"nstep": "3"}),
    "acrlpd": ("online", "acrlpd", {"horizon_length": "3"}),
    "acfql": ("off2on", "acfql", {"horizon_length": "3", "num_envs": "3"}),
    "qam": ("offline", "qam", {"horizon_length": "3"}),
    "qgf": ("offline", "qgf", {"horizon_length": "3"}),
}


@pytest.mark.parametrize("key", CASES)
def test_algorithms_train_on_a_dataset_whose_trajectories_stay_apart(key, tmp_path, monkeypatch):
    phase, name, flags = CASES[key]
    agents = []
    for cls in (OffPolicyAlgorithm, OfflineRLAlgorithm):
        init = cls.__init__
        monkeypatch.setattr(
            cls, "__init__",
            lambda self, *args, __init=init, **kwargs: (agents.append(self), __init(self, *args, **kwargs))[0] and None,
        )
    inputs = {
        "offline_dataset": str(write_dataset(tmp_path / "d.h5")), "offline_buffer_size": "64", "action_low": "-2",
        "action_high": "2", "total_timesteps": "48", "training_freq": "8", "learning_starts": "24",
        "num_offline_steps": "2", "num_online_steps": "24", "warmup_steps": "0",
        "online_replay_mode": "mixed", "offline_data_ratio": "0.5", "batch_size": "8",
        "save_final_checkpoint": False, **flags,
    }
    result = run_entry(PHASES[phase][0], build_argv(phase, name, tmp_path / "runs", inputs=inputs))
    assert result.returncode == 0, result.stderr[-1500:]
    agent = agents[-1]
    buffer = getattr(agent, "offline_replay_buffer", None) or agent.replay_buffer
    assert buffer.nstep == H and agent._global_update > 0
    # write_dataset: two 24-step trajectories, each ending in a truncation.
    flat = buffer.episode_ends[: buffer.size].T.flatten()
    ends = flat.nonzero().flatten().tolist()
    assert {23, 47} <= set(ends)
    stretch = 48 // buffer.num_envs
    assert set(ends) == {23, 47} | set(range(stretch - 1, 48, stretch))
    assert not buffer.dones[: buffer.size].any()


def _write_h5_with_success(path):
    """As ``_write_h5(..., "truncated")``; trajectory ``i`` succeeds at its last step iff ``i`` is even."""
    _write_h5(path, "truncated")
    with h5py.File(path, "a") as f:
        for i, length in enumerate(LENGTHS):
            f[f"traj_{i}"]["success"] = (np.arange(length) == length - 1) & (i % 2 == 0)
    return path


def _reference_return(traj: int, t: int, sparse: bool) -> float:
    if sparse and traj % 2:
        return -1.0 / (1.0 - GAMMA)
    return sum(GAMMA**j * (10.0 * traj + t + j + 1) for j in range(LENGTHS[traj] - t))


def _order_buffer(kind: str, num_envs: int):
    common = {"num_envs": num_envs, "buffer_size": num_envs * 40, "storage_device": "cpu", "sample_device": "cpu"}
    mc = {"gamma": GAMMA, "sparse_reward_mc": kind == "mc_sparse", "sparse_negative_reward": -1.0}
    if kind == "rebrac":
        return ReBRACReplayBuffer(OBS_SPACE, ACT_SPACE, **common)
    if kind == "sarsa":
        return SarsaMCReplayBuffer(OBS_SPACE, ACT_SPACE, **common, **mc)
    return MCReplayBuffer(OBS_SPACE, ACT_SPACE, **common, **mc)


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("kind", ["mc_dense", "mc_sparse", "sarsa", "rebrac"])
def test_returns_and_next_actions_follow_the_dataset_order(kind, num_envs, tmp_path):
    """Cal-QL computes return-to-go per trajectory of the flat dataset
    (3rd_party/Cal-QL/JaxCQL/replay_buffer.py:247-258), CORL's ReBRAC takes the next row's
    action (3rd_party/CORL/algorithms/offline/rebrac.py:224-246): whatever env column a row lands in."""
    buffer = _order_buffer(kind, num_envs)
    loaded = load_h5_dataset_to_replay_buffer(buffer, _write_h5_with_success(tmp_path / "d.h5"))
    per_env = loaded // num_envs
    rows = buffer._sampleable_rows() if kind == "rebrac" else buffer._valid_indices()
    sample = buffer._index_batch(rows[:, 0], rows[:, 1])
    first_row = {traj: sum(LENGTHS[:traj]) for traj in range(len(LENGTHS))}
    lost = 0
    if kind == "rebrac" and num_envs > 1:
        lost = num_envs  # each column's last row: its next action is not stored
    assert rows.shape[0] == loaded - lost
    for i in range(rows.shape[0]):
        traj, t = (int(v) for v in sample.obs["state"][i])
        flat_row = first_row[traj] + t
        if kind != "rebrac":
            assert float(sample.mc_returns[i]) == pytest.approx(_reference_return(traj, t, kind == "mc_sparse"), rel=1e-5)
        if kind in ("sarsa", "rebrac"):
            has_next = t + 1 < LENGTHS[traj]
            seam = (flat_row + 1) % per_env == 0
            if kind == "sarsa":
                assert bool(sample.next_action_valid[i]) == (has_next and not seam)
            if has_next and (kind == "rebrac" or not seam):
                assert sample.next_actions[i].tolist() == [traj, t + 1]


@pytest.mark.parametrize("kind", ["nstep", "chunked"])
def test_dataset_rows_never_continue_the_episode_running_in_the_buffer(kind, tmp_path):
    buffer = _buffer(kind, 1)
    for t in range(5):
        row = {"state": torch.tensor([[100.0, t]])}
        buffer.add(row, {"state": torch.tensor([[100.0, t + 1]])}, torch.zeros(1, 2), torch.ones(1), torch.zeros(1), torch.zeros(1))
    load_h5_dataset_to_replay_buffer(buffer, _write_h5(tmp_path / "d.h5", "truncated"))
    _, _, sample = _candidates(buffer)
    online = sample.obs["state"][:, 0] == 100.0
    assert online.any()
    assert (sample.next_obs["state"][online, 0] == 100.0).all()


@pytest.mark.parametrize("kind", ["nstep", "sarsa"])
def test_a_trajectory_split_over_two_loads_is_cut_at_the_split(kind):
    """Rows ``(0, t)`` of one 8-step trajectory, loaded as ``t < 5`` then ``t >= 5``."""
    buffer = _buffer("nstep", 1) if kind == "nstep" else _order_buffer("sarsa", 1)
    steps = torch.arange(9, dtype=torch.float32)
    obs = torch.stack((torch.zeros(9), steps), dim=1)
    ends = steps[:-1] == 7
    for part in (slice(0, 5), slice(5, 8)):
        rows, next_rows = {"state": obs[:-1][part]}, {"state": obs[1:][part]}
        add_flat_transitions(
            buffer, rows, next_rows, obs[:-1][part], torch.ones(8)[part], torch.zeros(8)[part],
            episode_ends=ends[part],
        )
    if kind == "nstep":
        _, _, sample = _candidates(buffer)
        reached = sample.next_obs["state"][:, 1] - sample.obs["state"][:, 1]
        assert reached[sample.obs["state"][:, 1] == 4].tolist() == [1.0]
        assert reached[sample.obs["state"][:, 1] == 3].tolist() == [2.0]
    else:
        rows = buffer._valid_indices()
        sample = buffer._index_batch(rows[:, 0], rows[:, 1])
        assert sample.obs["state"][:, 1].tolist() == [5.0, 6.0, 7.0]


def _rebrac_rows(rows: int, per_env: int, num_envs: int = 2) -> ReBRACReplayBuffer:
    """Row ``t`` of env ``e`` holds state ``(t, e)``."""
    buffer = ReBRACReplayBuffer(OBS_SPACE, ACT_SPACE, num_envs, per_env * num_envs, storage_device="cpu", sample_device="cpu")
    for t in range(rows):
        obs = {"state": torch.tensor([[float(t), float(e)] for e in range(num_envs)])}
        buffer.add(obs, obs, torch.zeros(num_envs, 2), torch.zeros(num_envs), torch.zeros(num_envs))
    return buffer


def _epoch_rows(buffer, draws: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in buffer.sample_without_repeat(1).obs["state"][0]) for _ in range(draws)]


def test_rebrac_epoch_drops_the_row_cut_after_its_permutation_was_drawn():
    buffer = _rebrac_rows(6, per_env=10)
    _epoch_rows(buffer, 2)
    buffer.cut_running_episodes()
    rows = _epoch_rows(buffer, 2 * buffer.epoch_size)
    assert buffer.epoch_size == 10
    assert not any(t == 5 for t, _ in rows)


def test_rebrac_epoch_covers_rows_again_once_a_full_lap_restores_pos_and_full():
    buffer = _rebrac_rows(4, per_env=4)
    buffer.cut_running_episodes()
    _epoch_rows(buffer, 1)
    for t in range(4):  # a whole lap: pos and full as before, no row cut any more
        obs = {"state": torch.tensor([[float(t), float(e)] for e in range(2)])}
        buffer.add(obs, obs, torch.zeros(2, 2), torch.zeros(2), torch.zeros(2))
    assert buffer.epoch_size == 8
    assert sorted(_epoch_rows(buffer, 8)) == [(t, e) for t in range(4) for e in range(2)]


def test_monte_carlo_returns_see_the_episode_end_in_the_rows_left_over():
    """One 5-step episode over 2 envs keeps 4 rows; the dropped 5th still ends it."""
    buffer = _order_buffer("mc_sparse", 2)
    steps = torch.arange(5, dtype=torch.float32)
    obs = {"state": torch.stack((torch.zeros(5), steps), dim=1)}
    rewards = steps + 1
    ends = steps == 4
    add_flat_transitions(buffer, obs, obs, torch.zeros(5, 2), rewards, torch.zeros(5), None, ends.float(), episode_ends=ends)
    rows = buffer._valid_indices()
    sample = buffer._index_batch(rows[:, 0], rows[:, 1])
    assert sorted(sample.obs["state"][:, 1].tolist()) == [0.0, 1.0, 2.0, 3.0]
    for t, mc in zip(sample.obs["state"][:, 1].tolist(), sample.mc_returns.tolist()):
        assert mc == pytest.approx(sum(GAMMA**j * (t + j + 1) for j in range(5 - int(t))), rel=1e-5)
