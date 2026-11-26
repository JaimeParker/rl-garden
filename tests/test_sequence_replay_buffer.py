"""Tests for ``rl_garden.buffers.sequence_replay_buffer.SequenceReplayBuffer``:
strict (``cross_episode=False``) mode never
returns a window crossing an episode boundary, but -- unlike the former
strict single-episode-window buffer it replaces -- no longer loses tail steps at an
episode's end; tolerant (``cross_episode=True``) mode's checkpoint/priority
scaffolding is reachable through a thin subclass and still returns
``episode_starts`` masks. Most of the tolerant-mode surface is already
covered end-to-end by ``test_recurrent_replay_buffer.py``/
``test_transformer_replay_buffer.py`` (unchanged by this refactor); this
file focuses on what's new: the unified constructor's mode dispatch and the
strict-mode tail-step fix.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.recurrent_replay_buffer import RecurrentReplayBuffer
from rl_garden.buffers.sequence_replay_buffer import SequenceReplayBuffer
from rl_garden.buffers.transformer_replay_buffer import TransformerReplayBuffer


def _obs_space() -> spaces.Dict:
    return spaces.Dict({"state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)})


def _dict_obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "rgb_front": spaces.Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _make_strict_buffer(
    obs_space=None, num_envs=1, per_env_buffer_size=8, horizon=2
) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        obs_space if obs_space is not None else _obs_space(),
        _action_space(),
        num_envs,
        per_env_buffer_size * num_envs,
        cross_episode=False,
        horizon=horizon,
        storage_device="cpu",
        sample_device="cpu",
    )


def _add_step(buf, t: int, *, done=None, episode_end=None, reward=None, next_obs=None):
    num_envs = buf.num_envs
    obs = {"state": torch.full((num_envs, 4), float(t))}
    if "rgb_front" in buf.observation_space.spaces:
        obs["rgb_front"] = torch.zeros((num_envs, 3, 8, 8), dtype=torch.uint8)
    if next_obs is None:
        next_obs = obs
    action = torch.zeros(num_envs, 2)
    reward = torch.zeros(num_envs) if reward is None else reward
    done = torch.zeros(num_envs) if done is None else done
    episode_end = torch.zeros(num_envs) if episode_end is None else episode_end
    buf.add(obs, next_obs, action, reward, done, episode_end)
    return obs


# ---------------------------------------------------------------------------
# Strict mode (cross_episode=False) -- construction/validation
# ---------------------------------------------------------------------------


def test_strict_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        _make_strict_buffer(horizon=0)


def test_strict_rejects_buffer_size_not_larger_than_horizon():
    with pytest.raises(ValueError):
        _make_strict_buffer(per_env_buffer_size=2, horizon=2)


def test_strict_rejects_tolerant_only_kwargs():
    with pytest.raises(ValueError):
        SequenceReplayBuffer(
            _obs_space(), _action_space(), 1, 8,
            cross_episode=False, horizon=2, stride=1,
        )


def test_priority_is_an_independent_constructor_argument():
    # priority: no longer derived
    # from cross_episode -- cross_episode=True now supports both
    # priority=True (tolerant, unchanged) and priority=False (uniform, new).
    tolerant = SequenceReplayBuffer(
        _obs_space(), _action_space(), 1, 8,
        cross_episode=True, priority=True,
        burn_in_len=1, learning_len=1, forward_len=1, stride=1,
    )
    assert tolerant.priority is True
    uniform = SequenceReplayBuffer(
        _obs_space(), _action_space(), 1, 8,
        cross_episode=True, priority=False, horizon=2,
    )
    assert uniform.priority is False
    strict = _make_strict_buffer()
    assert strict.priority is False


def test_cross_episode_false_priority_true_is_rejected():
    with pytest.raises(ValueError):
        SequenceReplayBuffer(
            _obs_space(), _action_space(), 1, 8,
            cross_episode=False, priority=True, horizon=2,
        )


def test_direct_instance_add_and_sample_unimplemented_for_tolerant_priority():
    buf = SequenceReplayBuffer(
        _obs_space(), _action_space(), 1, 8,
        cross_episode=True, priority=True,
        burn_in_len=1, learning_len=1, forward_len=1, stride=1,
        storage_device="cpu", sample_device="cpu",
    )
    with pytest.raises(NotImplementedError):
        buf.add({"state": torch.zeros(1, 4)}, {"state": torch.zeros(1, 4)},
                 torch.zeros(1, 2), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with pytest.raises(NotImplementedError):
        buf.sample(4)


# ---------------------------------------------------------------------------
# Strict mode -- never crosses an episode boundary
# ---------------------------------------------------------------------------


def test_sample_raises_before_any_valid_window_exists():
    buf = _make_strict_buffer(per_env_buffer_size=8, horizon=2)
    with pytest.raises(RuntimeError):
        buf.sample(batch_size=4)


def test_sampled_window_never_crosses_episode_boundary():
    # Two short episodes back to back: [0,1,2] (end at 2) and [3,4,5,6,7].
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    episode_ends = {2: 1.0}
    for t in range(8):
        _add_step(buf, t, episode_end=torch.tensor([episode_ends.get(t, 0.0)]))

    torch.manual_seed(0)
    for _ in range(50):
        sample = buf.sample(batch_size=16)
        obs_vals = sample.obs["state"]  # (horizon+1, B, 4)
        for b in range(obs_vals.shape[1]):
            ts = obs_vals[:, b, 0]
            assert (ts <= 2).all() or (ts >= 3).all()


def test_wraparound_does_not_produce_stale_window():
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=4, horizon=2)
    for t in range(10):
        _add_step(buf, t)

    torch.manual_seed(0)
    for _ in range(50):
        sample = buf.sample(batch_size=8)
        obs_vals = sample.obs["state"][:, :, 0]
        diffs = obs_vals[1:] - obs_vals[:-1]
        assert torch.all(diffs == 1.0)


def test_dict_obs_window_gather_with_image_key():
    buf = _make_strict_buffer(obs_space=_dict_obs_space(), num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(10):
        _add_step(buf, t)

    torch.manual_seed(0)
    sample = buf.sample(batch_size=8)
    assert isinstance(sample.obs, dict)
    assert sample.obs["state"].shape == (3, 8, 4)
    assert sample.obs["rgb_front"].shape == (3, 8, 3, 8, 8)


def test_sample_shapes_and_action_reward_alignment():
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=32, horizon=3)
    rewards_seq = [float(t) * 10.0 for t in range(20)]
    for t in range(20):
        _add_step(buf, t, reward=torch.tensor([rewards_seq[t]]))

    torch.manual_seed(0)
    sample = buf.sample(batch_size=16)
    assert sample.obs["state"].shape == (4, 16, 4)
    assert sample.action.shape == (3, 16, 2)
    assert sample.reward.shape == (3, 16)
    assert sample.terminated.shape == (3, 16)
    for b in range(16):
        t0 = int(sample.obs["state"][0, b, 0].item())
        expected = torch.tensor(rewards_seq[t0 : t0 + 3])
        assert torch.allclose(sample.reward[:, b], expected)


# ---------------------------------------------------------------------------
# Strict mode -- tail-step fix. Before this fix, ANY window whose
# span reached an episode's true final transition was rejected outright
# (a documented limitation of the buffer this replaces); SequenceReplayBuffer now
# accepts it and patches the true final observation in from the final-obs
# side table.
# ---------------------------------------------------------------------------


def test_strict_mode_accepts_a_window_ending_exactly_at_episode_end():
    # One 3-step episode [0, 1, 2] (episode_end at t=2), horizon=2: the only
    # window fully inside it is t0=0 -- but a window ending exactly at the
    # boundary, t0=1 (actions at steps 1 and 2, the episode's last), was
    # unconditionally rejected pre-fix.
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    true_final_obs = {"state": torch.full((1, 4), 999.0)}
    for t in range(3):
        is_end = t == 2
        _add_step(
            buf, t,
            episode_end=torch.tensor([1.0 if is_end else 0.0]),
            next_obs=true_final_obs if is_end else None,
        )
    # Pad a second, unrelated episode after so `buf.size` has more than one
    # candidate for the rejection-sampling loop to find.
    for t in range(3, 10):
        _add_step(buf, t)

    env = torch.zeros(1, dtype=torch.long)
    assert bool(buf._valid_window_batch(torch.tensor([1]), env).item())

    torch.manual_seed(0)
    found = False
    for _ in range(200):
        sample = buf.sample(batch_size=4)
        for b in range(4):
            ts = sample.obs["state"][:, b, 0]
            if ts[0].item() == 1.0:
                found = True
                # The window's last position must be the TRUE final obs
                # (999.0), not the next episode's aliased reset obs (3.0).
                assert sample.obs["state"][-1, b, 0].item() == 999.0
                assert bool(sample.terminated[-1, b].item()) is False
    assert found, "window t0=1 (ending at the episode boundary) was never sampled"


def test_strict_mode_tail_window_carries_a_true_termination():
    # Same as above, but the episode ends via a genuine termination (done),
    # not just truncation -- the returned `terminated` field's last position
    # must now be able to carry a positive example (documents the fix; the
    # old strict-mode buffer's module docstring called this impossible).
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(3):
        is_end = t == 2
        _add_step(
            buf, t,
            done=torch.tensor([1.0 if is_end else 0.0]),
            episode_end=torch.tensor([1.0 if is_end else 0.0]),
        )
    for t in range(3, 10):
        _add_step(buf, t)

    torch.manual_seed(0)
    saw_true_termination = False
    for _ in range(200):
        sample = buf.sample(batch_size=4)
        for b in range(4):
            if sample.obs["state"][0, b, 0].item() == 1.0 and bool(sample.terminated[-1, b].item()):
                saw_true_termination = True
    assert saw_true_termination


def test_window_needing_two_steps_past_episode_end_still_rejected():
    """The tail fix only accepts a window whose LAST position coincides with
    the boundary -- one needing real data strictly beyond it must still be
    rejected (strict mode never fabricates more than the one true final
    observation)."""
    buf = _make_strict_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(3):
        _add_step(buf, t, episode_end=torch.tensor([1.0 if t == 2 else 0.0]))
    env = torch.zeros(1, dtype=torch.long)
    # t0=2 is the episode's last step itself; a horizon=2 window from there
    # needs two more real steps past the boundary, which don't exist.
    assert not bool(buf._valid_window_batch(torch.tensor([2]), env).item())


# ---------------------------------------------------------------------------
# Tolerant mode (cross_episode=True) -- reachable through a thin subclass;
# checkpoint hooks + episode_starts masks round-trip through the unified
# base (full coverage stays in test_recurrent_replay_buffer.py /
# test_transformer_replay_buffer.py, unaffected by this refactor).
# ---------------------------------------------------------------------------


def test_tolerant_mode_returns_episode_starts_mask():
    buf = TransformerReplayBuffer(
        _obs_space(), _action_space(), 1, 32,
        burn_in_len=2, learning_len=2, forward_len=1,
        storage_device="cpu", sample_device="cpu",
    )
    for t in range(10):
        num_envs = buf.num_envs
        obs = {"state": torch.full((num_envs, 4), float(t))}
        episode_end = torch.tensor([1.0 if t == 4 else 0.0])
        buf.add(obs, obs, torch.zeros(num_envs, 2), torch.zeros(num_envs), torch.zeros(num_envs), episode_end)

    torch.manual_seed(0)
    sample = buf.sample(batch_size=4)
    assert sample.episode_starts.shape[1] == 4
    assert set(torch.unique(sample.episode_starts).tolist()) <= {0.0, 1.0}


def test_tolerant_mode_checkpoint_hooks_round_trip_hidden_state():
    buf = RecurrentReplayBuffer(
        _obs_space(), _action_space(), 1, 32,
        burn_in_len=2, learning_len=2, forward_len=1,
        rnn_type="gru", rnn_hidden_size=3, rnn_num_layers=1,
        storage_device="cpu", sample_device="cpu",
    )
    hidden = torch.full((1, 1, 3), 7.0)
    for t in range(10):
        obs = {"state": torch.full((1, 4), float(t))}
        buf.add(obs, obs, torch.zeros(1, 2), torch.zeros(1), torch.zeros(1), torch.zeros(1), hidden=hidden)
        hidden = hidden + 1.0

    torch.manual_seed(0)
    sample = buf.sample(batch_size=4)
    assert sample.initial_hidden_h.shape == (4, 1, 3)


# ---------------------------------------------------------------------------
# Uniform mode (cross_episode=True, priority=False) -- DreamerV3. No priority tree, no checkpoint grid: windows are drawn
# uniformly over ring-buffer-contiguous spans and may cross an episode
# boundary (masked downstream via the stored per-step is_first, not
# episode_starts). Every step's model carry is stored densely and can be
# written back after training.
# ---------------------------------------------------------------------------


def _make_uniform_buffer(
    obs_space=None, num_envs=1, per_env_buffer_size=16, horizon=3, carry_spec=None
) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        obs_space if obs_space is not None else _obs_space(),
        _action_space(),
        num_envs,
        per_env_buffer_size * num_envs,
        cross_episode=True,
        priority=False,
        horizon=horizon,
        carry_spec=carry_spec,
        storage_device="cpu",
        sample_device="cpu",
    )


def _add_uniform_step(buf, t: int, *, is_first=None, done=None, episode_end=None, carry=None):
    num_envs = buf.num_envs
    obs = {"state": torch.full((num_envs, 4), float(t))}
    action = torch.zeros(num_envs, 2)
    reward = torch.zeros(num_envs)
    done = torch.zeros(num_envs) if done is None else done
    episode_end = torch.zeros(num_envs) if episode_end is None else episode_end
    is_first = torch.zeros(num_envs) if is_first is None else is_first
    buf.add(obs, obs, action, reward, done, episode_end, is_first=is_first, carry=carry)


def test_uniform_requires_is_first_on_add():
    buf = _make_uniform_buffer(per_env_buffer_size=8, horizon=1)
    with pytest.raises(ValueError):
        buf.add(
            {"state": torch.zeros(1, 4)}, {"state": torch.zeros(1, 4)},
            torch.zeros(1, 2), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        )


def test_carry_spec_rejected_outside_uniform_mode():
    with pytest.raises(ValueError):
        SequenceReplayBuffer(
            _obs_space(), _action_space(), 1, 8,
            cross_episode=False, horizon=2, carry_spec={"deter": (2,)},
        )


def test_uniform_window_crossing_episode_boundary_carries_is_first_inside():
    buf = _make_uniform_buffer(per_env_buffer_size=32, horizon=3)
    # Episode 1: t=0,1,2 (truncated at t=2). Episode 2 starts at t=3.
    _add_uniform_step(buf, 0, is_first=torch.tensor([1.0]))
    _add_uniform_step(buf, 1)
    _add_uniform_step(buf, 2, episode_end=torch.tensor([1.0]))
    _add_uniform_step(buf, 3, is_first=torch.tensor([1.0]))
    for t in range(4, 8):
        _add_uniform_step(buf, t)

    torch.manual_seed(0)
    found = False
    for _ in range(100):
        batch = buf.sample(batch_size=8)
        row0 = batch.obs["state"][0, :, 0]
        cols = (row0 == 1.0).nonzero(as_tuple=False).flatten()  # t0=1 window: rows 1,2,3,4
        if cols.numel() == 0:
            continue
        found = True
        for c in cols.tolist():
            # loss rows are position 2, the final observation after it and position
            # 3 -- is_first is True only at position 3 (the new episode's first
            # observation), unshifted.
            assert batch.is_first[:, c].tolist() == [False, False, True]
    assert found, "a window starting at t0=1 (crossing the boundary) was never sampled"


def test_uniform_carry_round_trip():
    buf = _make_uniform_buffer(per_env_buffer_size=8, horizon=3, carry_spec={"deter": (2,)})
    for t in range(4):  # exactly horizon+1 steps written -> t0=0 is the only valid start
        _add_uniform_step(buf, t, carry={"deter": torch.full((1, 2), float(t))})

    batch = buf.sample(batch_size=6)
    assert batch.carry["deter"].shape == (6, 2)
    assert torch.allclose(batch.carry["deter"], torch.zeros(6, 2))  # t=0's stored carry


def test_uniform_write_back_carry_visible_on_next_sample():
    buf = _make_uniform_buffer(per_env_buffer_size=32, horizon=3, carry_spec={"deter": (2,)})
    for t in range(5):  # t0 in {0, 1} are valid
        _add_uniform_step(buf, t, carry={"deter": torch.full((1, 2), float(t))})

    torch.manual_seed(0)
    col = None
    for _ in range(50):
        batch = buf.sample(batch_size=16)
        row0 = batch.obs["state"][0, :, 0]
        cols = (row0 == 0.0).nonzero(as_tuple=False).flatten()
        if cols.numel() > 0:
            col = cols[0].item()
            idx_grid, env_grid = batch.indices
            marker = torch.full((idx_grid.shape[0], 1, 2), 555.0)
            buf.write_back_carry((idx_grid[:, col : col + 1], env_grid[:, col : col + 1]), {"deter": marker})
            break
    assert col is not None, "a t0=0 window was never sampled"

    torch.manual_seed(1)
    found = False
    for _ in range(50):
        batch2 = buf.sample(batch_size=16)
        row0 = batch2.obs["state"][0, :, 0]
        cols2 = (row0 == 1.0).nonzero(as_tuple=False).flatten()  # t0=1's row-0 carry is buffer position 1
        if cols2.numel() == 0:
            continue
        found = True
        assert torch.allclose(batch2.carry["deter"][cols2], torch.full((cols2.numel(), 2), 555.0))
        break
    assert found, "a t0=1 window was never sampled after the write-back"


# Two envs, episodes of 3 and 4 steps. The observation of step ``t`` of episode ``ep``
# in env ``e`` is (e, ep, t, 0); an episode's final observation has t == its length.
_EPISODE_LENGTHS = (3, 4)


def _episode_stream(steps: int, *, horizon: int, per_env: int = 32, terminate=False, **add_kwargs):
    buf = _make_uniform_buffer(
        num_envs=2, per_env_buffer_size=per_env, horizon=horizon, carry_spec={"deter": (1,)}
    )
    episode, step = [0, 0], [0, 0]
    for _ in range(steps):
        ended = torch.tensor([step[e] == _EPISODE_LENGTHS[e] - 1 for e in range(2)])

        def obs(dt):
            return {"state": torch.tensor([[e, episode[e], step[e] + dt, 0.0] for e in range(2)])}

        tag = torch.tensor([10.0 * episode[e] + step[e] + 1.0 for e in range(2)])
        buf.add(
            obs(0), obs(1), tag.unsqueeze(1).expand(2, 2).clone(), tag, ended & terminate, ended,
            is_first=torch.tensor([s == 0 for s in step]),
            carry={"deter": torch.zeros(2, 1)},
            **add_kwargs,
        )
        for e in range(2):
            episode[e], step[e] = (episode[e] + 1, 0) if ended[e] else (episode[e], step[e] + 1)
    return buf


def _reference_rows(env: int, steps: int):
    """The rows the official implementations store: ``(obs, action-in, reward, is_first, is_last)``."""
    rows, ep, t = [], 0, 0
    for _ in range(steps):
        tag = 10.0 * ep + t + 1.0
        rows.append(((env, ep, t), tag - 1.0 if t else 0.0, t == 0, False))
        if t == _EPISODE_LENGTHS[env] - 1:
            rows.append(((env, ep, t + 1), tag, False, True))
            ep, t = ep + 1, 0
        else:
            t += 1
    return rows


@pytest.mark.parametrize("terminate", [False, True])
def test_uniform_windows_are_the_windows_of_the_official_row_stream(terminate):
    steps, horizon = 14, 4
    buf = _episode_stream(steps, horizon=horizon, terminate=terminate)
    streams = [_reference_rows(env, steps) for env in range(2)]
    expected = {
        tuple(row[0] for row in streams[env][i : i + horizon + 1]): (env, i)
        for env in range(2)
        for i in range(len(streams[env]) - horizon)
    }
    torch.manual_seed(0)
    counts = dict.fromkeys(expected, 0)
    batch = buf.sample(6_000)
    for c in range(6_000):
        window = tuple(tuple(int(x) for x in row[:3]) for row in batch.obs["state"][:, c].tolist())
        counts[window] += 1  # KeyError: not a window of the official stream
        if counts[window] > 1:
            continue
        env, i = expected[window]
        for k, (_, incoming, is_first, is_last) in enumerate(streams[env][i + 1 : i + horizon + 1]):
            assert batch.reward[k, c] == incoming and (batch.action[k, c] == incoming).all()
            assert bool(batch.is_first[k, c]) is is_first
            assert bool(batch.is_last[k, c]) is is_last
            assert bool(batch.is_terminal[k, c]) is (is_last and terminate)
    # Every window, the ones starting at a final observation included, with equal probability
    # (6000 draws of 30-odd windows: five sigma is below 40 %).
    mean = 6_000 / len(counts)
    assert all(0.6 * mean < count < 1.4 * mean for count in counts.values()), counts


def test_uniform_write_back_carry_leaves_final_observation_rows_out():
    buf = _episode_stream(14, horizon=4)
    torch.manual_seed(0)
    batch = buf.sample(64)
    idx_grid, env_grid = batch.indices
    final = batch.is_last & ~batch.is_first
    assert final.any() and torch.equal(idx_grid == -1, final)
    buf.write_back_carry(batch.indices, {"deter": torch.full((4, 64, 1), 5.0)})
    written = torch.zeros(32, 2, dtype=torch.bool)
    written[idx_grid[~final], env_grid[~final]] = True
    assert torch.equal(buf._carry["deter"][..., 0] == 5.0, written)


def test_uniform_without_known_final_observations_keeps_the_ring_rows():
    buf = _episode_stream(14, horizon=4, final_obs_known=False)
    assert (buf._final_slot_ids == -1).all()
    batch = buf.sample(256)
    assert torch.equal(batch.is_last, batch.is_first)  # the reset row carries the ended transition
    assert (batch.indices[0] >= 0).all()
    assert ((batch.obs["state"][:, :, 2] < torch.tensor(_EPISODE_LENGTHS)[batch.obs["state"][0, :, 0].long()])).all()


def _old_file(buf: SequenceReplayBuffer, *, final_obs_reported: bool) -> dict:
    """What a uniform buffer saved before ``final_obs_known``: without a reported final
    observation every ended row kept the reset observation, the next ring row."""
    state = {k: v for k, v in buf.state_dict().items() if k != "final_obs_known_recorded"}
    if not final_obs_reported:
        table = {k: v.clone() for k, v in state["final_obs"].items()}
        for pos, env in (buf._final_slot_ids >= 0).nonzero().tolist():
            table["state"][buf._final_slot_ids[pos, env]] = buf.obs["state"][(pos + 1) % buf.per_env_buffer_size, env]
        state["final_obs"] = table
    return state


@pytest.mark.parametrize("per_env", [9, 32])
@pytest.mark.parametrize("final_obs_reported", [True, False])
def test_old_uniform_files_keep_only_true_final_observations(final_obs_reported, per_env):
    buf = _episode_stream(22, horizon=4, per_env=per_env)
    loaded = _make_uniform_buffer(num_envs=2, per_env_buffer_size=per_env, horizon=4, carry_spec={"deter": (1,)})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        loaded.load_state_dict(_old_file(buf, final_obs_reported=final_obs_reported))
    assert len(record) == (0 if final_obs_reported else 1)
    if final_obs_reported:
        assert torch.equal(loaded._final_slot_ids, buf._final_slot_ids)
        return
    assert (loaded._final_slot_ids == -1).all()
    assert len(loaded._free_final_slots) == loaded._next_final_slot
    batch = loaded.sample(256)
    assert torch.equal(batch.is_last, batch.is_first)


def test_files_with_recorded_final_observations_are_taken_as_they_are():
    buf = _episode_stream(22, horizon=4)
    state = {**_old_file(buf, final_obs_reported=False), "final_obs_known_recorded": True}
    loaded = _make_uniform_buffer(num_envs=2, per_env_buffer_size=32, horizon=4, carry_spec={"deter": (1,)})
    loaded.load_state_dict(state)
    assert torch.equal(loaded._final_slot_ids, buf._final_slot_ids)
    assert buf.state_dict()["final_obs_known_recorded"] is True


def test_uniform_final_observations_survive_the_ring_wrapping():
    steps, horizon = 40, 4
    buf = _episode_stream(steps, horizon=horizon, per_env=9)
    batch = buf.sample(2_000)
    streams = [[row[0] for row in _reference_rows(env, steps)] for env in range(2)]
    starts = set()
    for c in range(2_000):
        window = [tuple(int(x) for x in row[:3]) for row in batch.obs["state"][:, c].tolist()]
        stream = streams[window[0][0]]
        i = stream.index(window[0])
        assert window == stream[i : i + horizon + 1]
        starts.add((window[0][0], i))
    assert {env for env, _ in starts} == {0, 1}


def test_uniform_is_last_and_is_terminal_are_separate_for_truncation():
    buf = _make_uniform_buffer(per_env_buffer_size=8, horizon=1)
    _add_uniform_step(buf, 0, episode_end=torch.tensor([1.0]))  # truncation only: done stays False
    _add_uniform_step(buf, 1, is_first=torch.tensor([1.0]))

    batch = buf.sample(batch_size=32)  # the row after step 0 is its final observation or the reset
    assert batch.is_last.any() and batch.is_first.any()
    assert torch.equal(batch.is_last, ~batch.is_first)
    assert not batch.is_terminal.any()


def test_uniform_is_last_and_is_terminal_both_set_for_true_termination():
    buf = _make_uniform_buffer(per_env_buffer_size=8, horizon=1)
    _add_uniform_step(buf, 0, done=torch.tensor([1.0]), episode_end=torch.tensor([1.0]))
    _add_uniform_step(buf, 1, is_first=torch.tensor([1.0]))

    batch = buf.sample(batch_size=32)
    assert batch.is_last.any() and batch.is_first.any()
    assert torch.equal(batch.is_last, ~batch.is_first)
    assert torch.equal(batch.is_terminal, batch.is_last)
