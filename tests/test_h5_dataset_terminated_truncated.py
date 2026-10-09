"""Tests for the H5 loader's TD-``done`` vs. ``episode_end`` boundary split.

A WSRL-style H5 file writes ``terminated`` and ``truncated`` as two separate
datasets (``rl_garden/datasets/wsrl_generation.py``'s
``WSRLTrajectoryWriter.write_episode``). Before this fix,
``_transition_done`` ORed every terminal-like field -- including
``truncated``/``truncations`` -- into the TD ``done`` used for Bellman
bootstrapping, so an artificial ``max_episode_steps`` cutoff wrongly stopped
bootstrap the same way a true termination does. Online rollouts never do
this (``off_policy.py``'s ``_compute_done_masks``, default
``bootstrap_at_done="truncated"``, keeps bootstrapping through truncation),
so offline H5 loading disagreed with online training on the exact same
semantics -- a general bug, not specific to any one environment backend.

These tests pin:
  (a) a truncation boundary keeps TD ``done=0`` (bootstrap continues) while
      still closing out ``episode_end=1``; a true termination still sets
      ``done=1``.
  (b) the MC recursion resets at every true boundary (``episode_end``), not
      just at TD ``done`` -- including at a truncation that is *not* the
      trajectory's last step, which is the only arrangement that can
      actually distinguish "MC keyed on done" from "MC keyed on
      episode_end" (keying on either gives the same result at a
      trajectory's final index, since the backward recursion always starts
      from 0 there).
  (c) a truncation-only trajectory (no termination-like key at all) gets TD
      ``done=0`` everywhere and ``episode_end`` = the truncation flags --
      new fallback behavior for ``_load_traj_transitions`` (previously this
      shape wasn't distinguished from case (d) at all).
  (d) legacy ``dones``-only trajectories are unaffected: byte-identical to
      pre-fix behavior (this is the shape every existing
      ``test_h5_dataset_loader.py`` case already uses, with ``truncated``
      always absent or all-``False``).

Not retested here: explicit ``episode_end`` key priority over
terminated/truncated is already exercised by
``test_h5_dataset_loader.py::test_loader_uses_explicit_episode_end_separately_from_terminal``.
"""

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers import MCReplayBuffer, load_h5_dataset_to_replay_buffer
from rl_garden.buffers._dataset_common import _transition_done


def _make_buffer(num_envs: int, buffer_size: int, gamma: float = 0.9) -> MCReplayBuffer:
    return MCReplayBuffer(
        observation_space=spaces.Dict(
            {"state": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)}
        ),
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        num_envs=num_envs,
        buffer_size=buffer_size,
        gamma=gamma,
        storage_device="cpu",
        sample_device="cpu",
    )


# ---------------------------------------------------------------------------
# Direct unit tests of `_transition_done`.
# ---------------------------------------------------------------------------


def test_transition_done_excludes_truncation_from_td_done():
    traj = {
        "terminated": np.array([False, True, False]),
        "truncated": np.array([False, False, True]),
    }
    done, truncated = _transition_done(traj, 3, torch.device("cpu"))
    assert done.tolist() == [0.0, 1.0, 0.0]
    assert truncated is not None
    assert truncated.tolist() == [False, False, True]


def test_transition_done_truncation_only_is_not_last_step_fallback():
    """Only a truncation key exists (no terminated/terminations/done/dones):
    TD done must be all-False, not the degenerate last-step=1 fallback --
    the truncation key already carries a real (non-bootstrap-stopping)
    boundary signal."""
    traj = {"truncated": np.array([False, False, True])}
    done, truncated = _transition_done(traj, 3, torch.device("cpu"))
    assert done.tolist() == [0.0, 0.0, 0.0]
    assert truncated is not None
    assert truncated.tolist() == [False, False, True]


def test_transition_done_no_signal_at_all_falls_back_to_last_step():
    """Neither family of keys exists: keep the pre-existing degenerate
    fallback (mark only the final transition done)."""
    done, truncated = _transition_done({}, 3, torch.device("cpu"))
    assert done.tolist() == [0.0, 0.0, 1.0]
    assert truncated is None


# ---------------------------------------------------------------------------
# (a) Two-key trajectory: mid-episode success (terminated) + final timeout
#     (truncated, not terminated).
# ---------------------------------------------------------------------------


def test_terminated_and_truncated_keys_split_done_from_episode_end(tmp_path):
    path = tmp_path / "two_key.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((4, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((4, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((4, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.array([1, 2, 3, 4], dtype=np.float32))
        traj.create_dataset(
            "terminated", data=np.array([False, True, False, False])
        )
        traj.create_dataset(
            "truncated", data=np.array([False, False, False, True])
        )

    buffer = _make_buffer(num_envs=1, buffer_size=4)
    loaded = load_h5_dataset_to_replay_buffer(buffer, path)

    assert loaded == 4
    # Mid-episode success (idx 1): TD done=1.
    # Final timeout (idx 3, truncated & not terminated): TD done=0 (bootstrap
    # must continue through an artificial cutoff), but episode_end=1 (the
    # episode still closes there).
    assert buffer.dones[:4, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert buffer._episode_end[:4, 0].tolist() == [False, True, False, True]


# ---------------------------------------------------------------------------
# (b) MC recursion must reset at episode_end, not at TD done -- using a
#     truncation at a *non-final* index so the two boundaries actually
#     diverge somewhere the recursion can see it.
# ---------------------------------------------------------------------------


def test_mc_returns_reset_at_truncation_boundary_mid_trajectory(tmp_path):
    path = tmp_path / "mc_boundary.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((4, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((4, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((4, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.array([1, 2, 3, 4], dtype=np.float32))
        # Truncation at idx 1 (mid-trajectory, NOT the last step); true
        # termination at idx 3 (the last step).
        traj.create_dataset(
            "terminated", data=np.array([False, False, False, True])
        )
        traj.create_dataset(
            "truncated", data=np.array([False, True, False, False])
        )

    gamma = 0.9
    buffer = _make_buffer(num_envs=1, buffer_size=4, gamma=gamma)
    loaded = load_h5_dataset_to_replay_buffer(buffer, path)
    assert loaded == 4

    # TD done: only the true termination at idx 3 stops bootstrap; the
    # truncation at idx 1 does not.
    assert buffer.dones[:4, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
    # episode_end: both idx 1 (truncation) and idx 3 (termination) close an
    # episode -- this file packs two back-to-back episodes, [0, 1] and
    # [2, 3], into one `traj_0` group.
    assert buffer._episode_end[:4, 0].tolist() == [False, True, False, True]

    # Hand-computed MC returns, recursing backward and resetting at every
    # episode_end (not at every TD done):
    #   idx3: running = 4                                  -> 4.0
    #   idx2: running = 3 + 0.9*4                           -> 6.6
    #   idx1: running = 2 + 0.9*6.6*(1-1)=2  (reset: episode_end[1]=True)
    #   idx0: running = 1 + 0.9*2                           -> 2.8
    # If MC were (incorrectly) keyed on TD done=[0,0,0,1] instead, idx1
    # would NOT reset and would instead carry idx2/idx3's discounted reward
    # backward (running = 2 + 0.9*6.6 = 7.94), bleeding episode [2, 3]'s
    # return into episode [0, 1]. Since TD done and episode_end only differ
    # at idx 1 here (a non-final index), this is the only arrangement that
    # can tell the two keyings apart: at a trajectory's own last index, the
    # backward sweep always starts from a running total of 0, so the choice
    # of boundary flag there never changes the result.
    expected = torch.tensor([2.8, 2.0, 6.6, 4.0])
    assert torch.allclose(buffer._external_mc[:4, 0], expected)
    assert buffer._externally_valid[:4, 0].all()


# ---------------------------------------------------------------------------
# (c) Truncation-only data: no termination-like key at all.
# ---------------------------------------------------------------------------


def test_truncated_only_trajectory_has_zero_td_done_and_episode_end_from_truncation(
    tmp_path,
):
    path = tmp_path / "truncated_only.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((3, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((3, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((3, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.ones(3, dtype=np.float32))
        traj.create_dataset("truncated", data=np.array([False, False, True]))

    buffer = _make_buffer(num_envs=1, buffer_size=4)
    loaded = load_h5_dataset_to_replay_buffer(buffer, path)

    assert loaded == 3
    assert buffer.dones[:3, 0].tolist() == [0.0, 0.0, 0.0]
    assert buffer._episode_end[:3, 0].tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# (d) Legacy `dones`-only data: regression guard, unchanged from pre-fix.
# ---------------------------------------------------------------------------


def test_dones_only_legacy_trajectory_is_unchanged(tmp_path):
    path = tmp_path / "legacy_dones.h5"
    with h5py.File(path, "w") as f:
        traj = f.create_group("traj_0")
        traj.create_dataset("obs", data=np.zeros((4, 2), dtype=np.float32))
        traj.create_dataset("next_obs", data=np.ones((4, 2), dtype=np.float32))
        traj.create_dataset("actions", data=np.zeros((4, 1), dtype=np.float32))
        traj.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        traj.create_dataset("dones", data=np.array([False, False, False, True]))

    gamma = 0.9
    buffer = _make_buffer(num_envs=1, buffer_size=4, gamma=gamma)
    loaded = load_h5_dataset_to_replay_buffer(buffer, path)

    assert loaded == 4
    assert buffer.dones[:4, 0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert buffer._episode_end[:4, 0].tolist() == [False, False, False, True]
    expected = torch.tensor([3.439, 2.71, 1.9, 1.0])
    assert torch.allclose(buffer._external_mc[:4, 0], expected, atol=1e-4)
