from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.episode_slice_buffer import EpisodeSliceBuffer


def _obs_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _dict_obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "rgb": spaces.Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _make_buffer(
    obs_space=None,
    num_envs=1,
    per_env_buffer_size=8,
    horizon=2,
) -> EpisodeSliceBuffer:
    return EpisodeSliceBuffer(
        obs_space if obs_space is not None else _obs_space(),
        _action_space(),
        num_envs,
        per_env_buffer_size * num_envs,
        horizon=horizon,
        storage_device="cpu",
        sample_device="cpu",
    )


def _add_step(buf: EpisodeSliceBuffer, t: int, *, done=None, episode_end=None, reward=None):
    num_envs = buf.num_envs
    if buf._is_dict_obs:
        obs = {
            "state": torch.full((num_envs, 4), float(t)),
            "rgb": torch.zeros((num_envs, 3, 8, 8), dtype=torch.uint8),
        }
    else:
        obs = torch.full((num_envs, 4), float(t))
    action = torch.zeros(num_envs, 2)
    reward = torch.zeros(num_envs) if reward is None else reward
    done = torch.zeros(num_envs) if done is None else done
    episode_end = torch.zeros(num_envs) if episode_end is None else episode_end
    buf.add(obs, obs, action, reward, done, episode_end)


def test_rejects_non_positive_horizon():
    with pytest.raises(ValueError):
        _make_buffer(horizon=0)


def test_rejects_buffer_size_not_larger_than_horizon():
    with pytest.raises(ValueError):
        _make_buffer(per_env_buffer_size=2, horizon=2)


def test_sample_raises_before_any_valid_window_exists():
    buf = _make_buffer(per_env_buffer_size=8, horizon=2)
    with pytest.raises(RuntimeError):
        buf.sample(batch_size=4)


def test_sampled_window_never_crosses_episode_boundary():
    # Two short episodes back to back: [0,1,2] (end at 2) and [3,4,5,6,7].
    buf = _make_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    episode_ends = {2: 1.0}
    for t in range(8):
        _add_step(buf, t, episode_end=torch.tensor([episode_ends.get(t, 0.0)]))

    torch.manual_seed(0)
    for _ in range(50):
        sample = buf.sample(batch_size=16)
        # obs values were written as `t` itself, so every window's obs values
        # reveal exactly which physical positions were gathered; a boundary
        # crossing would show up as values straddling {0,1,2} and {3,4,...}.
        obs_vals = sample.obs  # (horizon+1, B, 4)
        for b in range(obs_vals.shape[1]):
            ts = obs_vals[:, b, 0]
            assert (ts <= 2).all() or (ts >= 3).all()


def test_window_starting_at_last_step_of_short_episode_is_never_valid():
    """The episode [0,1,2] (ends at 2) is only 3 steps long; a horizon=2
    window needs 3 contiguous steps of the SAME episode ahead of t0, so t0=2
    (the last step of that episode) must never be selected, since the next
    physical slot holds the following episode's data."""
    buf = _make_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(8):
        _add_step(buf, t, episode_end=torch.tensor([1.0 if t == 2 else 0.0]))

    env = torch.zeros(1, dtype=torch.long)
    assert not bool(buf._valid_window_batch(torch.tensor([2]), env).item())
    # But a window starting well inside the second episode is valid.
    assert bool(buf._valid_window_batch(torch.tensor([3]), env).item())


def test_wraparound_does_not_produce_stale_window():
    # buffer_size=4 forces a wraparound after 4 writes into a single ongoing
    # episode (no episode_end at all) -- a window whose gathered indices span
    # the physical wrap-around discontinuity must be rejected.
    buf = _make_buffer(num_envs=1, per_env_buffer_size=4, horizon=2)
    for t in range(10):
        _add_step(buf, t)

    torch.manual_seed(0)
    for _ in range(50):
        sample = buf.sample(batch_size=8)
        obs_vals = sample.obs[:, :, 0]  # (horizon+1, B)
        # Every gathered window must be exactly consecutive integers (the
        # values we wrote), i.e. genuinely temporally contiguous.
        diffs = obs_vals[1:] - obs_vals[:-1]
        assert torch.all(diffs == 1.0)


def test_dict_obs_window_gather_matches_box_obs_semantics():
    buf = _make_buffer(obs_space=_dict_obs_space(), num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(10):
        _add_step(buf, t)

    torch.manual_seed(0)
    sample = buf.sample(batch_size=8)
    assert isinstance(sample.obs, dict)
    assert sample.obs["state"].shape == (3, 8, 4)
    assert sample.obs["rgb"].shape == (3, 8, 3, 8, 8)
    diffs = sample.obs["state"][1:, :, 0] - sample.obs["state"][:-1, :, 0]
    assert torch.all(diffs == 1.0)


def test_sample_shapes_and_action_reward_alignment():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=32, horizon=3)
    rewards_seq = [float(t) * 10.0 for t in range(20)]
    for t in range(20):
        _add_step(buf, t, reward=torch.tensor([rewards_seq[t]]))

    torch.manual_seed(0)
    sample = buf.sample(batch_size=16)
    assert sample.obs.shape == (4, 16, 4)
    assert sample.action.shape == (3, 16, 2)
    assert sample.reward.shape == (3, 16)
    assert sample.terminated.shape == (3, 16)
    # reward[i] must be the reward received transitioning obs[i] -> obs[i+1].
    for b in range(16):
        t0 = int(sample.obs[0, b, 0].item())
        expected = torch.tensor(rewards_seq[t0 : t0 + 3])
        assert torch.allclose(sample.reward[:, b], expected)


def test_terminated_flag_is_never_true_within_a_sampled_window():
    """Documents the known limitation from the module docstring: since a
    window ending on the true terminal transition is always rejected, the
    `terminated` field returned by sample() never carries a positive
    example."""
    buf = _make_buffer(num_envs=1, per_env_buffer_size=32, horizon=2)
    for t in range(20):
        is_end = t % 5 == 4
        _add_step(
            buf,
            t,
            done=torch.tensor([1.0 if is_end else 0.0]),
            episode_end=torch.tensor([1.0 if is_end else 0.0]),
        )

    torch.manual_seed(0)
    for _ in range(20):
        sample = buf.sample(batch_size=16)
        assert not sample.terminated.any()
