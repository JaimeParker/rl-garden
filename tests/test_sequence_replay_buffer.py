"""Tests for ``rl_garden.buffers.sequence_replay_buffer.SequenceReplayBuffer``
(model-based-base plan 1.6): strict (``cross_episode=False``) mode never
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


def test_priority_is_derived_from_cross_episode():
    # priority is not a constructor kwarg -- it is fully determined by
    # cross_episode (model-based-base plan 1.6 review fix).
    tolerant = SequenceReplayBuffer(
        _obs_space(), _action_space(), 1, 8,
        cross_episode=True,
        burn_in_len=1, learning_len=1, forward_len=1, stride=1,
    )
    assert tolerant.priority is True
    strict = _make_strict_buffer()
    assert strict.priority is False


def test_direct_instance_add_and_sample_unimplemented_for_cross_episode_true():
    buf = SequenceReplayBuffer(
        _obs_space(), _action_space(), 1, 8,
        cross_episode=True,
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
# Strict mode -- tail-step fix (model-based-base plan 1.6 / scratchpad
# tdmpc2-upstream-vs-port.md section 3). Before this fix, ANY window whose
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
