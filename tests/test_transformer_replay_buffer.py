from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers.transformer_replay_buffer import TransformerReplayBuffer
from rl_garden.common.obs_utils import index_obs


def _obs_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _make_buffer(
    num_envs=1,
    per_env_buffer_size=8,
    burn_in_len=2,
    learning_len=2,
    forward_len=1,
    gamma=1.0,
) -> TransformerReplayBuffer:
    return TransformerReplayBuffer(
        _obs_space(),
        _action_space(),
        num_envs,
        per_env_buffer_size * num_envs,
        burn_in_len=burn_in_len,
        learning_len=learning_len,
        forward_len=forward_len,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )


def _add_step(buf: TransformerReplayBuffer, t: int, *, done=None, episode_end=None, reward=None):
    num_envs = buf.num_envs
    obs = torch.full((num_envs, 4), float(t))
    action = torch.zeros(num_envs, 2)
    reward = torch.zeros(num_envs) if reward is None else reward
    done = torch.zeros(num_envs) if done is None else done
    episode_end = torch.zeros(num_envs) if episode_end is None else episode_end
    buf.add(obs, obs, action, reward, done, episode_end)


def test_add_has_no_hidden_parameter():
    """TransformerReplayBuffer never stores a per-transition hidden checkpoint
    -- add() must not accept a `hidden` kwarg at all (the key structural
    difference from RecurrentReplayBuffer.add())."""
    buf = _make_buffer()
    with pytest.raises(TypeError):
        buf.add(
            torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 2), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), hidden=torch.zeros(1, 3),
        )


def test_every_position_is_checkpoint_aligned_not_just_burn_in_len_multiples():
    """Key behavioral difference from RecurrentReplayBuffer: stride=1 means
    every position becomes its own checkpoint slot, so window starts are NOT
    constrained to burn_in_len-periodic positions."""
    buf = _make_buffer(num_envs=1, per_env_buffer_size=8, burn_in_len=2, learning_len=2, forward_len=1)
    for t in range(8):
        _add_step(buf, t)

    assert buf._current_ckpt_pos[0].item() == 8
    for pos in range(8):
        assert buf._ckpt_slot_to_pos[pos, 0].item() == pos

    # A position that is NOT a multiple of burn_in_len (e.g. pos=3) must still
    # be a valid, ready window start once enough contiguous data follows it.
    env = torch.zeros(1, dtype=torch.long)
    t0 = torch.tensor([3])
    assert bool(buf._valid_window_batch(t0, env).item())


def test_sample_draws_from_non_aligned_positions():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=32, burn_in_len=2, learning_len=2, forward_len=1)
    for t in range(32):
        _add_step(buf, t)

    generator = torch.Generator().manual_seed(0)
    sample = buf.sample(batch_size=200, generator=generator)
    t0s = buf._ckpt_slot_to_pos[
        sample.priority_indices % buf.checkpoint_capacity,
        sample.priority_indices // buf.checkpoint_capacity,
    ]
    # burn_in_len=2 -- if sampling were still constrained to a stride-2 grid,
    # every t0 would be even. With stride=1 we expect at least one odd t0.
    assert (t0s % 2 != 0).any()


def test_nstep_window_matches_manual_accumulation():
    gamma = 0.9
    buf = _make_buffer(
        num_envs=1, per_env_buffer_size=8, burn_in_len=1, learning_len=2, forward_len=2, gamma=gamma
    )
    rewards_seq = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for t in range(8):
        _add_step(buf, t, reward=torch.tensor([rewards_seq[t]]))

    t0 = torch.tensor([0])
    env = torch.zeros(1, dtype=torch.long)
    rewards, discounts = buf._accumulate_nstep_window(t0, env)
    expected_0 = rewards_seq[1] + gamma * rewards_seq[2]
    expected_1 = rewards_seq[2] + gamma * rewards_seq[3]
    assert torch.allclose(rewards[0, 0], torch.tensor(expected_0))
    assert torch.allclose(rewards[1, 0], torch.tensor(expected_1))
    assert torch.allclose(discounts[0, 0], torch.tensor(gamma**2))
    assert torch.allclose(discounts[1, 0], torch.tensor(gamma**2))


def test_episode_starts_reset_within_sampled_window():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=8, burn_in_len=1, learning_len=2, forward_len=1)
    _add_step(buf, 0, episode_end=torch.ones(1))
    for t in range(1, 8):
        _add_step(buf, t)

    t0 = torch.tensor([1])
    env = torch.zeros(1, dtype=torch.long)
    idx_grid, env_grid = buf._gather_window(t0, env)
    starts = (buf._ep_relative_step[idx_grid, env_grid] == 0).float()
    assert starts[0, 0].item() == 1.0
    assert torch.all(starts[1:, 0] == 0.0)


def test_sample_never_reads_uninitialized_data():
    buf = _make_buffer(num_envs=2, per_env_buffer_size=16, burn_in_len=2, learning_len=2, forward_len=1)
    for t in range(16):
        _add_step(buf, t)

    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        sample = buf.sample(batch_size=8, generator=generator)
        assert torch.isfinite(sample.rewards).all()
        assert torch.isfinite(sample.discounts).all()


def test_sample_has_no_hidden_state_fields():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=8, burn_in_len=2, learning_len=2, forward_len=1)
    for t in range(8):
        _add_step(buf, t)
    sample = buf.sample(batch_size=4, generator=torch.Generator().manual_seed(0))
    assert not hasattr(sample, "initial_hidden_h")
    assert not hasattr(sample, "initial_hidden_c")


def test_priority_update_shifts_sampling_distribution():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=16, burn_in_len=2, learning_len=2, forward_len=1)
    for t in range(16):
        _add_step(buf, t)

    generator = torch.Generator().manual_seed(0)
    baseline = buf.sample(batch_size=200, generator=generator)
    baseline_counts = torch.bincount(baseline.priority_indices, minlength=buf.checkpoint_capacity)

    target_leaf = baseline.priority_indices[0].reshape(1)
    buf.update_priorities(target_leaf, torch.tensor([1000.0]))

    generator2 = torch.Generator().manual_seed(1)
    boosted = buf.sample(batch_size=200, generator=generator2)
    boosted_counts = torch.bincount(boosted.priority_indices, minlength=buf.checkpoint_capacity)

    assert boosted_counts[target_leaf.item()] > baseline_counts[target_leaf.item()]


def test_sample_raises_before_any_checkpoint_ready():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=8, burn_in_len=2, learning_len=2, forward_len=1)
    with pytest.raises(RuntimeError):
        buf.sample(batch_size=4)


def test_truncation_bootstrap_uses_patched_final_obs_not_reset_obs():
    buf = _make_buffer(num_envs=1, per_env_buffer_size=8, burn_in_len=1, learning_len=1, forward_len=1)
    true_final_obs = torch.full((1, 4), 999.0)
    for t in range(8):
        if t == 1:
            action = torch.zeros(1, 2)
            reward = torch.zeros(1)
            obs = torch.full((1, 4), float(t))
            buf.add(
                obs, true_final_obs, action, reward,
                torch.zeros(1), torch.ones(1),  # done=False, episode_end=True
            )
        else:
            _add_step(buf, t)

    t0 = torch.tensor([0])
    env = torch.zeros(1, dtype=torch.long)
    idx_grid, env_grid = buf._gather_window(t0, env)
    window_obs = index_obs(buf.obs, (idx_grid, env_grid))
    patched = buf._patch_final_obs(window_obs, idx_grid, env_grid)
    assert torch.allclose(patched[2, 0], true_final_obs[0])
