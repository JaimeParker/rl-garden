"""``cut_running_episodes()``: an env reset outside ``env.step`` ends what the
replay buffer was recording.

Each buffer class stores rows whose content says which env, episode and step they
belong to. Before and after the cut, every sample the buffer can return is
compared with a reference computed from that record.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect

import pytest
import torch

try:
    from tests._cli import run_entry
    from tests.test_entrypoint_builds import (
        IMAGE_ENV_ID,
        LOCOMOTION_NAMED_PENDULUM_ID,
        PHASES,
        _registered,
        build_argv,
        cpu_only_and_restored_rng,
        inputs_for,
        shared_inputs,
    )
    from tests.test_replay_buffer_state import (
        ACT_SPACE,
        BUFFERS,
        CARRY_SPEC,
        CPU,
        EPISODE_LENGTHS,
        FRAME_STACK,
        NUM_ENVS,
        OBS_SPACE,
        PER_ENV,
        STACKED_SPACE,
        WINDOW,
        assert_tree_equal,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry
    from test_entrypoint_builds import (
        IMAGE_ENV_ID,
        LOCOMOTION_NAMED_PENDULUM_ID,
        PHASES,
        _registered,
        build_argv,
        cpu_only_and_restored_rng,
        inputs_for,
        shared_inputs,
    )
    from test_replay_buffer_state import (
        ACT_SPACE,
        BUFFERS,
        CARRY_SPEC,
        CPU,
        EPISODE_LENGTHS,
        FRAME_STACK,
        NUM_ENVS,
        OBS_SPACE,
        PER_ENV,
        STACKED_SPACE,
        WINDOW,
        assert_tree_equal,
    )

from rl_garden.algorithms.off_policy import OffPolicyAlgorithm
from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.buffers.dataset_common import add_flat_transitions
from rl_garden.buffers.mc_buffer import MCReplayBuffer, MCReplayBufferMixin
from rl_garden.buffers.memory_efficient_buffer import MemoryEfficientReplayBuffer
from rl_garden.buffers.nstep_buffer import LazyNextNStepReplayBuffer, NStepReplayBuffer
from rl_garden.buffers.nstep_sampling import NStepSamplingMixin
from rl_garden.buffers.recurrent_replay_buffer import RecurrentReplayBuffer
from rl_garden.buffers.rehearsal_sequence_buffer import RehearsalSequenceBuffer
from rl_garden.buffers.sarsa_buffer import SarsaMCReplayBuffer
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.buffers.state_sequence_replay_buffer import StateSequenceReplayBuffer
from rl_garden.common.checkpoint import (
    load_replay_buffer_file,
    load_replay_buffer_state_dict,
    replay_buffer_state_dict,
    save_replay_buffer_file,
)

__all__ = ["cpu_only_and_restored_rng", "shared_inputs"]  # fixtures

GAMMA = 0.9
SEQUENCE_GAMMA = 0.99  # the sequence buffers' default
NEGATIVE_MC = -1.0 / (1.0 - GAMMA)
HISTORY_LEN = 3


# --- recorded stream -------------------------------------------------------------------------


def state_of(env, ep, t):
    return torch.tensor([float(env), float(ep), float(t)])


def pixel_of(env, ep, t):
    return 128 * env + 1 + 11 * ep + t


def action_of(ep, t):
    return torch.tensor([float(ep), float(t)])


def reward_of(ep, t):
    return 100.0 * ep + t + 1.0


def tag_of(ep, t):
    return 10.0 * ep + t


def ends_terminal(ep):
    return ep % 2 == 1


def is_success(ep, t):
    return ep % 2 == 0 and t == 2


class Stream:
    """Feeds one buffer and remembers which step each stored row holds."""

    def __init__(self, buffer) -> None:
        self.buffer = buffer
        self.rows: list[list[tuple[int, int]]] = []  # per step, per env: (ep, t)
        self.current = [[0, 0] for _ in range(NUM_ENVS)]
        self.ended: set[tuple[int, int]] = set()
        self.cut_episodes: set[tuple[int, int]] = set()
        self.lengths: dict[tuple[int, int], int] = {}

    @property
    def recording(self):
        return getattr(self.buffer, "online", self.buffer)

    def _obs(self, dt: int):
        buffer, steps = self.recording, [(env, ep, t + dt) for env, (ep, t) in enumerate(self.current)]
        if isinstance(buffer, MemoryEfficientReplayBuffer):
            frames = [
                [pixel_of(env, ep, max(t - back, 0)) for back in range(FRAME_STACK - 1, -1, -1)]
                for env, ep, t in steps
            ]
            image = torch.tensor(frames, dtype=torch.uint8).reshape(NUM_ENVS, FRAME_STACK, 1, 1, 1)
            image = image.expand(NUM_ENVS, FRAME_STACK, 4, 4, 3)
        else:
            image = torch.tensor([pixel_of(*step) for step in steps], dtype=torch.uint8)
            image = image.reshape(NUM_ENVS, 1, 1, 1).expand(NUM_ENVS, 4, 4, 3)
        return {"state": torch.stack([state_of(*step) for step in steps]), "rgb_cam": image}

    def add(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._add_one()

    def _add_one(self) -> None:
        buffer = self.recording
        ends = torch.tensor(
            [t == EPISODE_LENGTHS[env] - 1 for env, (_, t) in enumerate(self.current)]
        )
        done = torch.tensor(
            [bool(ends[env]) and ends_terminal(ep) for env, (ep, _) in enumerate(self.current)]
        )
        tags = torch.tensor([tag_of(ep, t) for ep, t in self.current])
        extras = {
            "episode_end": ends,
            "is_first": torch.tensor([t == 0 for _, t in self.current]),
            "carry": {
                key: tags.reshape(-1, *[1] * len(shape)).expand(NUM_ENVS, *shape).clone()
                for key, shape in CARRY_SPEC.items()
            },
            "success": torch.tensor([is_success(ep, t) for ep, t in self.current]),
            "hidden": None,
        }
        if isinstance(buffer, RecurrentReplayBuffer):
            hidden = tags.reshape(1, -1, 1).expand(1, NUM_ENVS, buffer.rnn_hidden_size)
            hidden = hidden.to(buffer.storage_device, copy=True)
            extras["hidden"] = (hidden, hidden) if buffer._has_cell_state else hidden
        accepted = set(inspect.signature(buffer.add).parameters)
        if isinstance(buffer, SequenceReplayBuffer) and not buffer.cross_episode:
            accepted = {"episode_end"}
        elif isinstance(buffer, MCReplayBufferMixin):
            accepted = {"episode_end", "success"}
        self.buffer.add(
            self._obs(0),
            self._obs(1),
            torch.stack([action_of(ep, t) for ep, t in self.current]),
            torch.tensor([reward_of(ep, t) for ep, t in self.current]),
            done if buffer.dones.dtype == torch.bool else done.float(),
            **{name: value for name, value in extras.items() if name in accepted},
        )
        self.rows.append([(ep, t) for ep, t in self.current])
        for env, (ep, t) in enumerate(self.current):
            self.lengths[(env, ep)] = t + 1
            if bool(ends[env]):
                self.ended.add((env, ep))
                self.current[env] = [ep + 1, 0]
            else:
                self.current[env][1] += 1

    def reset_envs(self) -> None:
        self.buffer.cut_running_episodes()
        for env, (ep, t) in enumerate(self.current):
            if t > 0:
                self.cut_episodes.add((env, ep))
                self.current[env] = [ep + 1, 0]

    # --- reference ---

    @property
    def first_step(self) -> int:
        return max(0, len(self.rows) - PER_ENV)

    def stored(self):
        """``(position, env, step, ep, t)`` of every stored row."""
        for step in range(self.first_step, len(self.rows)):
            for env in range(NUM_ENVS):
                yield (step % PER_ENV, env, step, *self.rows[step][env])

    def steps_by_position(self) -> dict[int, int]:
        return {step % PER_ENV: step for step in range(self.first_step, len(self.rows))}

    def is_end(self, step, env) -> bool:
        ep, t = self.rows[step][env]
        return (env, ep) in self.ended and t == self.lengths[(env, ep)] - 1

    def is_cut(self, step, env) -> bool:
        ep, t = self.rows[step][env]
        return (env, ep) in self.cut_episodes and t == self.lengths[(env, ep)] - 1

    def is_terminal(self, step, env) -> bool:
        return self.is_end(step, env) and ends_terminal(self.rows[step][env][0])


# --- every sample a buffer can return ---------------------------------------------------------


def _patched(buffer, name, value, call):
    setattr(buffer, name, value)
    try:
        return call()
    finally:
        delattr(buffer, name)


def candidate_grid(buffer):
    t0 = torch.arange(buffer.size).repeat_interleave(buffer.num_envs)
    return t0, torch.arange(buffer.num_envs).repeat(buffer.size)


def all_samples(buffer):
    """``(t0, env, sample)`` over every start the buffer accepts; ``None`` if it has none."""
    if isinstance(buffer, RehearsalSequenceBuffer):
        buffer = buffer.online
    if isinstance(buffer, MCReplayBufferMixin):
        t0, env = buffer._valid_indices().unbind(1)
        return (t0, env, buffer._index_batch(t0, env)) if t0.numel() else None
    t0, env = candidate_grid(buffer)
    if isinstance(buffer, NStepSamplingMixin):
        valid, name = buffer._valid_nstep_batch(t0, env), "_sample_valid_indices"
    elif isinstance(buffer, MemoryEfficientReplayBuffer):
        valid, name = buffer._valid_batch(t0, env), "_sample_valid_indices"
    elif isinstance(buffer, SequenceReplayBuffer) and buffer.priority:
        leaves = torch.arange(buffer.checkpoint_capacity * buffer.num_envs)
        slot, env = leaves % buffer.checkpoint_capacity, leaves // buffer.checkpoint_capacity
        t0 = buffer._ckpt_slot_to_pos[slot, env]
        valid = (t0 >= 0) & buffer._valid_window_batch(t0.clamp(min=0), env)
        t0, env, leaves = t0[valid], env[valid], leaves[valid]
        if not t0.numel():
            return None
        starts = (t0, env, leaves, torch.ones(t0.numel()))
        sample = _patched(
            buffer, "_sample_valid_window_starts", lambda n, generator=None: starts,
            lambda: buffer.sample(t0.numel()),
        )
        return t0, env, sample
    elif isinstance(buffer, SequenceReplayBuffer) and buffer.cross_episode:
        valid, name = buffer._valid_uniform_window_batch(t0, env), "_sample_uniform_window_starts"
    elif isinstance(buffer, SequenceReplayBuffer):
        valid, name = buffer._valid_window_batch(t0, env), "_sample_valid_window_starts"
    else:
        return t0, env, buffer._index_batch(t0, env)
    t0, env = t0[valid], env[valid]
    if not t0.numel():
        return None
    starts = (t0, env)
    if name == "_sample_uniform_window_starts":
        starts = (t0, env, torch.zeros_like(t0, dtype=torch.bool))
    sample = _patched(buffer, name, lambda *args, **kwargs: starts, lambda: buffer.sample(t0.numel()))
    return t0, env, sample


def _by_start(buffer):
    drawn = all_samples(buffer)
    if drawn is None:
        return None, {}
    t0, env, sample = drawn
    return sample, {(int(t), int(e)): i for i, (t, e) in enumerate(zip(t0, env))}


def _close(actual, expected):
    assert torch.allclose(torch.as_tensor(actual, dtype=torch.float32),
                          torch.as_tensor(expected, dtype=torch.float32)), (actual, expected)


# --- per-class references ---------------------------------------------------------------------


def check_single_transitions(stream):
    sample, index = _by_start(stream.buffer)
    assert len(index) == (len(stream.rows) - stream.first_step) * NUM_ENVS
    for pos, env, step, ep, t in stream.stored():
        i = index[(pos, env)]
        assert torch.equal(sample.obs["state"][i], state_of(env, ep, t))
        assert torch.equal(sample.next_obs["state"][i], state_of(env, ep, t + 1))
        assert bool(sample.dones[i]) == stream.is_terminal(step, env)


def check_rebrac(stream):
    """``next_actions`` is the action of the following ring row (documented on
    ``ReBRACReplayBufferSample``), so it is the one field here that does reach over an episode end
    and the write position; the row before a cut is not sampled. In tree only the offline ReBRAC
    samples this buffer."""
    check_single_transitions(stream)
    sample, index = _by_start(stream.buffer)
    by_position = stream.steps_by_position()
    for pos, env, *_ in stream.stored():
        following = by_position.get((pos + 1) % PER_ENV)
        expected = torch.zeros(2) if following is None else action_of(*stream.rows[following][env])
        assert torch.equal(sample.next_actions[index[(pos, env)]], expected)


def check_nstep(stream):
    buffer = stream.buffer
    lazy, n = isinstance(buffer, LazyNextNStepReplayBuffer), buffer.nstep
    sample, index = _by_start(buffer)
    expected = {}
    for pos, env, _, ep, t in stream.stored():
        length, ended = stream.lengths[(env, ep)], (env, ep) in stream.ended
        cut = (env, ep) in stream.cut_episodes
        if lazy and cut:
            length -= 1  # the last transition has no stored successor
        usable = length - t
        if (ended or cut) and usable > 0 or usable >= n + (lazy and not ended):
            expected[(pos, env)] = (ep, t, min(n, usable), ended and ends_terminal(ep) and usable <= n)
    assert index.keys() == expected.keys()
    for start, (ep, t, k, terminal) in expected.items():
        i, env = index[start], start[1]
        assert torch.equal(sample.obs["state"][i], state_of(env, ep, t))
        assert torch.equal(sample.next_obs["state"][i], state_of(env, ep, t + k))
        _close(sample.rewards[i], sum(GAMMA**j * reward_of(ep, t + j) for j in range(k)))
        assert bool(sample.dones[i]) == terminal
        _close(sample.discounts[i], 0.0 if terminal else GAMMA**k)
        if isinstance(buffer, ChunkedReplayBuffer):
            assert sample.valid[i].tolist() == [j < k for j in range(n)]
            for j in range(k):
                assert torch.equal(sample.actions[i, j], action_of(ep, t + j))
        elif isinstance(buffer, NStepReplayBuffer):
            assert buffer._valid_nstep(*start)
            reward, discount, next_obs = buffer._compute_nstep(*start)
            _close(reward, sample.rewards[i])
            _close(discount, sample.discounts[i])
            if not lazy:
                assert torch.equal(next_obs["state"], sample.next_obs["state"][i])


def check_memory_efficient(stream):
    sample, index = _by_start(stream.buffer)
    expected = {
        (pos, env): (step, ep, t)
        for pos, env, step, ep, t in stream.stored()
        if step - min(t, FRAME_STACK - 1) >= stream.first_step
    }
    assert index.keys() == expected.keys()
    backs = range(FRAME_STACK - 1, -1, -1)
    for start, (step, ep, t) in expected.items():
        i, env = index[start], start[1]
        assert sample.obs["rgb_cam"][i, :, 0, 0, 0].tolist() == [
            pixel_of(env, ep, max(t - back, 0)) for back in backs]
        assert sample.next_obs["rgb_cam"][i, :, 0, 0, 0].tolist() == [
            pixel_of(env, ep, max(t + 1 - back, 0)) for back in backs]
        assert torch.equal(sample.next_obs["state"][i], state_of(env, ep, t + 1))
        assert bool(sample.dones[i]) == stream.is_terminal(step, env)


def check_strict_sequence(stream):
    horizon = stream.buffer.horizon
    sample, index = _by_start(stream.buffer)
    expected = {}
    for pos, env, step, ep, t in stream.stored():
        length = stream.lengths[(env, ep)]
        reach = length if (env, ep) in stream.ended else length - 1
        if t + horizon <= reach:
            expected[(pos, env)] = (step, ep, t)
    assert index.keys() == expected.keys()
    for start, (step, ep, t) in expected.items():
        i, env = index[start], start[1]
        for j in range(horizon + 1):
            assert torch.equal(sample.obs["state"][j, i], state_of(env, ep, t + j))
        for j in range(horizon):
            assert torch.equal(sample.action[j, i], action_of(ep, t + j))
            _close(sample.reward[j, i], reward_of(ep, t + j))
            assert bool(sample.terminated[j, i]) == stream.is_terminal(step + j, env)
    if isinstance(stream.buffer, StateSequenceReplayBuffer):
        check_history(stream)


def check_history(stream):
    stored = list(stream.stored())
    draws = iter([torch.tensor([row[0] for row in stored]), torch.tensor([row[1] for row in stored])])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "randint", lambda *args, **kwargs: next(draws))
        sample = stream.buffer.sample_history(len(stored), HISTORY_LEN)
    for i, (_, env, step, ep, t) in enumerate(stored):
        behind = min(t, step - stream.first_step)
        for row in range(HISTORY_LEN):
            back = HISTORY_LEN - 1 - row
            assert torch.equal(sample.obs["state"][row, i], state_of(env, ep, t - min(back, behind)))
            assert bool(sample.obs_valid[row, i]) == (back <= behind)
        previous = action_of(ep, t - 1) if behind else torch.zeros(2)
        assert torch.equal(sample.prev_action[i], previous)


def check_uniform_sequence(stream):
    buffer = stream.recording
    horizon = buffer.horizon
    sample, index = _by_start(buffer)

    def window(step, env):
        """``(step, is_final)`` of the ``horizon + 1`` rows from ``step`` on, an ended
        episode's final observation being a row; ``None`` across a cut or the newest row."""
        rows = [(step, False)]
        while len(rows) <= horizon:
            step, is_final = rows[-1]
            if stream.is_end(step, env) and not is_final:
                rows.append((step, True))
            elif step + 1 < len(stream.rows) and not stream.is_cut(step, env):
                rows.append((step + 1, False))
            else:
                return None
        return rows

    expected = {(pos, env): window(step, env) for pos, env, step, _, _ in stream.stored()}
    expected = {start: rows for start, rows in expected.items() if rows is not None}
    assert index.keys() == expected.keys()
    for start, rows in expected.items():
        i, env = index[start], start[1]
        for j, (step, is_final) in enumerate(rows):
            ep, t = stream.rows[step][env]
            assert torch.equal(sample.obs["state"][j, i], state_of(env, ep, t + is_final))
        for key in CARRY_SPEC:
            assert (sample.carry[key][i] == tag_of(*stream.rows[rows[0][0]][env])).all()
        for j, ((left, left_final), (arrived, final)) in enumerate(zip(rows, rows[1:])):
            assert bool(sample.is_first[j, i]) == (not final and stream.rows[arrived][env][1] == 0)
            assert bool(sample.is_last[j, i]) == final
            assert bool(sample.terminated[j, i]) == (final and stream.is_terminal(left, env))
            if left_final:
                assert (sample.action[j, i] == 0).all() and sample.reward[j, i] == 0
            else:
                assert torch.equal(sample.action[j, i], action_of(*stream.rows[left][env]))
                _close(sample.reward[j, i], reward_of(*stream.rows[left][env]))


def check_tolerant_sequence(stream):
    buffer = stream.buffer
    sample, index = _by_start(buffer)
    by_position = stream.steps_by_position()
    checkpoints = set()
    for env in range(NUM_ENVS):
        for pos in buffer._ckpt_slot_to_pos[:, env].tolist():
            if pos < 0:
                continue
            step = by_position[pos]
            assert stream.rows[step][env][1] % buffer.stride == 0
            checkpoints.add((pos, env))
    newest = len(stream.rows) - 1
    for env in range(NUM_ENVS):
        ep, t = stream.rows[newest][env]
        if t % buffer.stride == 0:
            assert (newest % PER_ENV, env) in checkpoints
    last_learning = buffer.burn_in_len + buffer.learning_len - 1
    expected = {}
    for pos, env in checkpoints:
        step = by_position[pos]
        read = 1  # rows of the window the losses read
        while read < buffer.window_len and not (
            read > last_learning and stream.is_end(step + read - 1, env)
        ):
            if step + read > newest or stream.is_cut(step + read - 1, env):
                break
            read += 1
        else:
            expected[(pos, env)] = read
    assert index.keys() == expected.keys()
    for (pos, env), i in index.items():
        step = by_position[pos]
        if hasattr(sample, "initial_hidden_h"):
            assert (sample.initial_hidden_h[i] == tag_of(*stream.rows[step][env])).all()
        for j in range(expected[(pos, env)]):
            ep, t = stream.rows[step + j][env]
            assert bool(sample.episode_starts[j, i]) == (t == 0)
            assert torch.equal(sample.obs["state"][j, i], state_of(env, ep, t))
        for j in range(buffer.learning_len):
            first = step + buffer.burn_in_len + j
            assert torch.equal(sample.actions[j, i], action_of(*stream.rows[first][env]))
            reward, discount, final = 0.0, 1.0, None
            for row in range(first, first + buffer.forward_len):
                reward += discount * reward_of(*stream.rows[row][env])
                discount *= SEQUENCE_GAMMA
                if stream.is_end(row, env):
                    if stream.is_terminal(row, env):
                        discount = 0.0
                    else:
                        final = row
                    break
            _close(sample.rewards[j, i], reward)
            _close(sample.discounts[j, i], discount)
            k = int(sample.final_index[j, i])
            assert (k >= 0) == (final is not None)
            if final is not None:
                ep, t = stream.rows[final][env]
                assert torch.equal(sample.final_obs["state"][k], state_of(env, ep, t + 1))
                assert int(sample.final_steps[k]) == final - step - buffer.burn_in_len
                assert int(sample.final_columns[k]) == i
    assert set(sample.final_index[sample.final_index >= 0].tolist()) == set(
        range(sample.final_steps.numel())
    )


def check_mc(stream):
    buffer = stream.buffer
    sample, index = _by_start(buffer)
    expected = {
        (pos, env): (step, ep, t)
        for pos, env, step, ep, t in stream.stored()
        if (env, ep) in stream.ended
    }
    assert index.keys() == expected.keys()
    assert buffer.sampleable_size == len(expected)
    for start, (step, ep, t) in expected.items():
        i, env = index[start], start[1]
        value, succeeded, length = 0.0, False, stream.lengths[(env, ep)]
        for later in range(length - 1, t - 1, -1):
            succeeded |= is_success(ep, later)
            value = reward_of(ep, later) + GAMMA * value
            if buffer.sparse_reward_mc and not succeeded:
                value = NEGATIVE_MC
        _close(sample.mc_returns[i], value)
        assert bool(sample.dones[i]) == stream.is_terminal(step, env)
        if isinstance(buffer, SarsaMCReplayBuffer):
            assert bool(sample.next_action_valid[i]) == (t < length - 1)
            if t < length - 1:
                assert torch.equal(sample.next_actions[i], action_of(ep, t + 1))
    if isinstance(buffer, SarsaMCReplayBuffer):
        for pos, env, step, _, _ in stream.stored():
            row = buffer._index_batch(torch.tensor([pos]), torch.tensor([env]))
            if stream.is_cut(step, env):
                assert not bool(row.next_action_valid)


CHECKS = {
    "plain": check_single_transitions,
    "grasp_penalty": check_single_transitions,
    "rebrac": check_rebrac,
    "mc": check_mc,
    "mc_sparse": check_mc,
    "sarsa_mc": check_mc,
    "memory_efficient": check_memory_efficient,
    "nstep": check_nstep,
    "lazy_nstep": check_nstep,
    "chunked": check_nstep,
    "strict_sequence": check_strict_sequence,
    "state_sequence": check_strict_sequence,
    "uniform_sequence": check_uniform_sequence,
    "rehearsal": check_uniform_sequence,
    "recurrent_lstm": check_tolerant_sequence,
    "recurrent_gru": check_tolerant_sequence,
    "transformer": check_tolerant_sequence,
}


def new_buffer(name):
    if name == "chunked":
        return ChunkedReplayBuffer(
            OBS_SPACE, ACT_SPACE, NUM_ENVS, NUM_ENVS * PER_ENV, horizon_length=3, gamma=GAMMA, **CPU
        )
    return BUFFERS[name]()


def reloaded(buffer, name, path):
    save_replay_buffer_file(path, buffer)
    target = new_buffer(name)
    load_replay_buffer_file(path, target)
    return target


def state_copy(buffer):
    return copy.deepcopy(replay_buffer_state_dict(buffer))


def test_rebrac_does_not_sample_the_row_before_a_cut():
    """CORL's ReBRAC keeps no transition without its own next action
    (3rd_party/CORL/algorithms/offline/rebrac.py:224, the dataset's last row)."""
    stream = Stream(new_buffer("rebrac"))
    stream.add(20)
    stream.reset_envs()
    stream.add(1)
    check_rebrac(stream)
    assert stream.is_cut(19, 0)
    rows = stream.buffer._sampleable_rows()
    assert rows.shape[0] == 21 * NUM_ENVS - NUM_ENVS
    assert not (rows[:, 0] == 19).any()
    assert stream.buffer.epoch_size == rows.shape[0]
    assert stream.buffer.sample(8).next_actions.shape == (8, 2)


def test_every_buffer_class_is_checked():
    assert CHECKS.keys() == BUFFERS.keys()


# ReBRAC is here for its single-row fields; the row before a cut is not sampled.
# Env 0 (7-step episodes) ends exactly at 21 and 28 rows; 24 and 48 leave the cursor at 0 of a full
# buffer.
@pytest.mark.parametrize("steps", [20, 21, 24, 28, 48, 53])
@pytest.mark.parametrize("reload", [True, False], ids=["resumed", "second_learn"])
@pytest.mark.parametrize("name", CHECKS)
def test_no_sample_spans_the_cut(name, reload, steps, tmp_path):
    stream = Stream(new_buffer(name))
    check = CHECKS[name]
    stream.add(steps)
    check(stream)
    if reload:
        stream.buffer = reloaded(stream.buffer, name, tmp_path / "before.pt")
    stream.reset_envs()
    check(stream)
    for step in range(12):
        stream.add()
        check(stream)
        if step == 2:  # short enough that windows can hold two cuts
            stream.reset_envs()
            check(stream)


@pytest.mark.parametrize("name", CHECKS)
def test_cut_state_survives_a_reload(name, tmp_path):
    stream = Stream(new_buffer(name))
    stream.add(20)
    stream.reset_envs()
    copied = reloaded(stream.buffer, name, tmp_path / "after.pt")
    assert_tree_equal(state_copy(stream.buffer), state_copy(copied))
    other = copy.deepcopy(stream)
    other.buffer = copied
    for each in (stream, other):
        each.add(10)
        CHECKS[name](each)
    assert_tree_equal(state_copy(stream.buffer), state_copy(copied))


@pytest.mark.parametrize("name", CHECKS)
def test_cut_is_idempotent_and_leaves_ended_rows_flags_and_rng_alone(name):
    stream = Stream(new_buffer(name))
    empty = state_copy(stream.buffer)
    rng = torch.get_rng_state()
    stream.reset_envs()
    assert_tree_equal(empty, state_copy(stream.buffer))

    stream.add(20)
    before = state_copy(stream.buffer)
    stream.reset_envs()
    after = state_copy(stream.buffer)
    for key in ("obs", "actions", "rewards", "dones", "pos", "full"):
        assert_tree_equal(before[key], after[key])
    stream.reset_envs()
    assert_tree_equal(after, state_copy(stream.buffer))
    assert torch.equal(rng, torch.get_rng_state())

    if name == "rebrac":
        return  # it stores no episode ends: a cut always stops its last row's next action
    ended = Stream(new_buffer(name))
    ended.add(63)  # 7 and 9 step episodes both end at row 62
    before = state_copy(ended.buffer)
    ended.reset_envs()
    assert_tree_equal(before, state_copy(ended.buffer))


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(CHECKS.keys() - {"lazy_nstep"}))  # CPU storage only
def test_cut_of_a_cuda_buffer_matches_the_cpu_one_and_uses_no_random_numbers(name, monkeypatch):
    if torch.cuda.device_count() == 0:
        pytest.skip("needs CUDA")

    def cut_state(check_rng):
        stream = Stream(new_buffer(name))
        stream.add(20)
        before = (torch.get_rng_state(), torch.cuda.get_rng_state()) if check_rng else ()
        stream.reset_envs()
        after = (torch.get_rng_state(), torch.cuda.get_rng_state()) if check_rng else ()
        assert all(map(torch.equal, before, after))
        stream.add(5)
        return {k: v for k, v in state_copy(stream.buffer).items() if not k.endswith("_device")}

    def on_cpu(tree):
        if isinstance(tree, dict):
            return {key: on_cpu(value) for key, value in tree.items()}
        return tree.cpu() if isinstance(tree, torch.Tensor) else tree

    expected = cut_state(check_rng=False)
    monkeypatch.setitem(CPU, "storage_device", "cuda")
    monkeypatch.setitem(CPU, "sample_device", "cuda")
    cuda = cut_state(check_rng=True)
    assert cuda["rewards"].device.type == "cuda"
    assert_tree_equal(expected, on_cpu(cuda))


def test_lazy_cut_of_a_one_step_episode_drops_only_that_row():
    stream = Stream(new_buffer("lazy_nstep"))
    stream.add(15)  # env 0 is one step into its third episode
    stream.reset_envs()
    stream.add(5)
    check_nstep(stream)


def test_mc_file_without_the_cut_column_loads():
    source = Stream(new_buffer("mc"))
    source.add(20)
    state = state_copy(source.buffer)
    del state["episode_cut"]
    target = copy.deepcopy(source)
    target.buffer = new_buffer("mc")
    load_replay_buffer_state_dict(target.buffer, state)
    check_mc(target)
    target.reset_envs()
    target.add(10)
    check_mc(target)


# --- mmap -------------------------------------------------------------------------------------


def _mmap_buffer(name, directory, mode):
    common = {"mmap_dir": directory, "mmap_mode": mode, **CPU}
    size = (ACT_SPACE, NUM_ENVS, NUM_ENVS * PER_ENV)
    if name == "nstep":
        return NStepReplayBuffer(OBS_SPACE, *size, nstep=3, gamma=GAMMA, **common)
    return MemoryEfficientReplayBuffer(
        STACKED_SPACE, *size, image_keys=["rgb_cam"], frame_stack=FRAME_STACK, **common)


@pytest.mark.parametrize("steps", [20, 24, 53])
@pytest.mark.parametrize("name", ["nstep", "memory_efficient"])
def test_mmap_cut_is_written_to_disk(name, steps, tmp_path):
    stream = Stream(_mmap_buffer(name, tmp_path / "replay", "create"))
    stream.add(steps)
    stream.buffer.flush()

    stream.buffer = _mmap_buffer(name, tmp_path / "replay", "open")
    stream.reset_envs()
    stream.buffer.flush()
    state = state_copy(stream.buffer)

    stream.buffer = _mmap_buffer(name, tmp_path / "replay", "open")
    assert_tree_equal(state, state_copy(stream.buffer))
    for _ in range(10):
        stream.add()
        CHECKS[name](stream)


# --- a dataset followed by online rows (off2on append) ----------------------------------------


def _load_dataset(buffer, *, mc_returns):
    """Two 13-step trajectories; the second one's last step carries no end marker."""
    obs = torch.stack([state_of(9, ep, t) for ep in range(2) for t in range(13)])
    next_obs = torch.stack([state_of(9, ep, t + 1) for ep in range(2) for t in range(13)])
    image = torch.zeros(26, 4, 4, 3, dtype=torch.uint8)
    ends = torch.zeros(26, dtype=torch.bool)
    ends[12] = True
    add_flat_transitions(
        buffer,
        {"state": obs, "rgb_cam": image},
        {"state": next_obs, "rgb_cam": image},
        torch.ones(26, 2),
        torch.arange(26.0),
        torch.zeros(26),
        mc_returns=mc_returns,
        episode_ends=ends,
    )


def _online_rows(buffer, count, *, end_last):
    for t in range(count):
        ends = torch.full((buffer.num_envs,), end_last and t == count - 1)
        obs = {"state": torch.full((buffer.num_envs, 3), 50.0 + t),
               "rgb_cam": torch.zeros(buffer.num_envs, 4, 4, 3, dtype=torch.uint8)}
        next_obs = {**obs, "state": obs["state"] + 1}
        buffer.add(obs, next_obs, torch.full((buffer.num_envs, 2), 5.0),
                   torch.full((buffer.num_envs,), 1000.0), torch.zeros(buffer.num_envs),
                   episode_end=ends)


def test_append_keeps_the_loaded_returns_and_starts_online_rows_afresh():
    buffer = MCReplayBuffer(OBS_SPACE, ACT_SPACE, NUM_ENVS, 80, gamma=GAMMA, **CPU)
    returns = torch.arange(26.0) * 3
    _load_dataset(buffer, mc_returns=returns)
    buffer.cut_running_episodes()
    _online_rows(buffer, 4, end_last=True)
    t0, env, sample = all_samples(buffer)
    assert t0.numel() == 26 + 8
    dataset = t0 < 13
    assert torch.equal(sample.mc_returns[dataset], returns[env[dataset] * 13 + t0[dataset]])
    _close(sample.mc_returns[~dataset],
           [sum(1000.0 * GAMMA**j for j in range(4 - (int(t) - 13))) for t in t0[~dataset]])


def test_append_without_loaded_returns_never_completes_the_dataset_tail():
    buffer = MCReplayBuffer(OBS_SPACE, ACT_SPACE, 1, 40, gamma=GAMMA, **CPU)
    _load_dataset(buffer, mc_returns=None)
    buffer.cut_running_episodes()
    _online_rows(buffer, 4, end_last=True)
    t0, _, _ = all_samples(buffer)
    assert t0.tolist() == [*range(13), *range(26, 30)]


def test_append_keeps_every_chunk_of_the_dataset_tail():
    buffer = ChunkedReplayBuffer(OBS_SPACE, ACT_SPACE, 1, 40, horizon_length=3, gamma=GAMMA, **CPU)
    _load_dataset(buffer, mc_returns=None)
    buffer.cut_running_episodes()
    _online_rows(buffer, 4, end_last=False)
    t0, _, sample = all_samples(buffer)
    assert t0.tolist() == [*range(26), 26, 27]
    tail = {int(t): i for i, t in enumerate(t0)}
    for t in (24, 25):
        i = tail[t]
        assert sample.valid[i].tolist() == [j < 26 - t for j in range(3)]
        assert torch.equal(sample.next_obs["state"][i], state_of(9, 1, 13))
        assert not bool(sample.dones[i])


# --- every algorithm family, resumed by its command line ---------------------------------------


@dataclasses.dataclass
class Family:
    name: str
    flags: dict
    phase: str = "online"
    buffer: type = NStepReplayBuffer
    #: False when the algorithm refuses to save its replay buffer.
    saved: bool = True


MEMORY_EFFICIENT = {"env_id": IMAGE_ENV_ID, "obs.rgb": "front", "obs.frame_stack": "2"}
OFF2ON = {
    "env_id": LOCOMOTION_NAMED_PENDULUM_ID, "online_replay_mode": "append", "warmup_steps": "0",
    "num_offline_steps": "2", "num_online_steps": "40",
}
WINDOW_FLAGS = {key: str(value) for key, value in WINDOW.items()}
FAMILIES = {
    "sac": Family("sac", {"nstep": "3"}),
    "flash_sac": Family("flash_sac", {"n_step": "3"}),
    "drqv2": Family("drqv2", {}),
    "drqv2_lazy": Family("drqv2", {"replay_lazy_next_obs": True}, buffer=LazyNextNStepReplayBuffer, saved=False),
    "rlpd": Family("rlpd", {"nstep": "3"}),
    "acrlpd": Family("acrlpd", {"horizon_length": "3"}, buffer=ChunkedReplayBuffer),
    "rlpd_hybrid": Family("rlpd_hybrid", MEMORY_EFFICIENT, buffer=MemoryEfficientReplayBuffer),
    "recurrent_sac": Family("recurrent_sac", WINDOW_FLAGS, buffer=RecurrentReplayBuffer),
    "transformer_sac": Family(
        "transformer_sac", {**WINDOW_FLAGS, "memory_len": "4"}, buffer=SequenceReplayBuffer),
    "tdmpc2": Family("tdmpc2", {"horizon": "2"}, buffer=SequenceReplayBuffer),
    "supermimic": Family("supermimic", {}, buffer=StateSequenceReplayBuffer),
    "dreamer_v3": Family("dreamer_v3", {}, buffer=SequenceReplayBuffer),
    "ncrl": Family("ncrl", {"total_frames": "40"}, buffer=RehearsalSequenceBuffer, saved=False),
    "calql": Family("calql", {}, phase="off2on", buffer=MCReplayBuffer),
    "acfql": Family("acfql", {"horizon_length": "3"}, phase="off2on", buffer=ChunkedReplayBuffer),
}


def _storages(buffer):
    trees = [buffer.obs, getattr(buffer, "next_obs", None), buffer.actions, buffer.rewards]
    leaves = []
    for tree in trees:
        if tree is not None:
            leaves += list(getattr(tree, "data", {}).values()) if not isinstance(tree, torch.Tensor) else [tree]
    return leaves


def _comparable(sample, keep) -> dict[str, torch.Tensor]:
    fields = {f.name: getattr(sample, f.name) for f in dataclasses.fields(sample)}
    batch_dim = 1 if "episode_starts" in fields or "terminated" in fields else 0
    out = {}
    if "valid" in fields:
        fields["actions"] = fields["actions"] * fields["valid"].unsqueeze(-1)
    for name in ("obs", "next_obs", "actions", "action", "rewards", "reward", "mc_returns"):
        value = fields.get(name)
        if isinstance(value, dict):
            out.update({f"{name}/{k}": v.index_select(batch_dim, keep) for k, v in value.items()})
        elif value is not None:
            dim = 0 if name == "mc_returns" else batch_dim
            out[name] = value.index_select(dim, keep)
    return out


def assert_rows_never_mix(buffer, boundary):
    """Samples that start before row ``boundary`` do not depend on the rows after it, and the
    other way round."""
    buffer = getattr(buffer, "online", buffer)
    for changed, kept in (
        (slice(boundary, buffer.size), lambda t0: t0 < boundary),
        (slice(0, boundary), lambda t0: t0 >= boundary),
    ):
        other = copy.deepcopy(buffer)
        if hasattr(other, "_mc_table"):
            other._mc_table = None
        for leaf in _storages(other):
            leaf[changed] = leaf[changed] * 0 + 7
        t0, _, original = all_samples(buffer)
        changed_t0, _, altered = all_samples(other)
        assert torch.equal(t0, changed_t0)
        keep = kept(t0).nonzero().flatten()
        assert keep.numel() > 0
        assert_tree_equal(_comparable(original, keep), _comparable(altered, keep))


@pytest.mark.parametrize("key", FAMILIES)
def test_resumed_training_never_joins_episodes(
    key, tmp_path, shared_inputs, monkeypatch
):
    family = FAMILIES[key]
    phase, name = family.phase, family.name
    learn = OffPolicyAlgorithm.learn
    started: list[int] = []
    agents = []

    def learn_twice(self, total_timesteps):
        """Two ``learn()`` calls, each starting in the middle of the episodes."""
        agents.append(self)
        middle = self._global_step + (total_timesteps - self._global_step) // 2
        for target in (middle, total_timesteps):
            started.append(self.replay_buffer.pos)
            learn(self, target)
        return self

    monkeypatch.setattr(OffPolicyAlgorithm, "learn", learn_twice)
    if family.buffer is MemoryEfficientReplayBuffer:  # a field of downstream args classes
        args_cls = _registered(phase)[name].args_cls
        monkeypatch.setattr(args_cls, "memory_efficient_buffer", True, raising=False)
    inputs = {
        "learning_starts": "1000000", "total_timesteps": "40", "training_freq": "4",
        "save_final_checkpoint": False, **inputs_for(phase, name, shared_inputs, tmp_path),
        **(OFF2ON if phase == "off2on" else {}), **family.flags,
    }

    def run(**extra):
        argv = build_argv(phase, name, tmp_path / "runs", inputs={**inputs, **extra})
        result = run_entry(PHASES[phase][0], argv)
        assert result.returncode == 0, result.stderr[-1500:]

    run()
    if family.saved:
        checkpoint = agents[0].save(tmp_path / "checkpoint_1.pt", include_replay_buffer=True)
        run(load_checkpoint=str(checkpoint), load_replay_buffer=True,
            total_timesteps="80", num_offline_steps="0")
        assert agents[-1] is not agents[0]
    buffer = agents[-1].replay_buffer
    assert isinstance(buffer, family.buffer)
    boundaries = [row for row in started if row > 0]
    assert len(boundaries) == (1 if not family.saved else 4 if phase == "off2on" else 3)
    assert boundaries[-1] < buffer.size and not buffer.full
    for boundary in boundaries:
        assert_rows_never_mix(buffer, boundary)
    if isinstance(buffer, RecurrentReplayBuffer):
        for boundary in boundaries:
            slots = (buffer._ckpt_slot_to_pos == boundary).nonzero()
            assert len(slots) == buffer.num_envs
            for slot, env in slots.tolist():
                assert not buffer.hidden_checkpoints_h[slot, env].any()
    online = getattr(buffer, "online", buffer)
    if hasattr(online, "_is_first"):
        assert all(bool(online._is_first[boundary].all()) for boundary in boundaries)
