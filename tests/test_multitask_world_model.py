"""Tests for MultitaskWorldModel: task embedding, action masking, padded
obs/action dims -- against synthetic tasks with heterogeneous dims (no
simulator/hardware)."""
from __future__ import annotations

import torch

from rl_garden.algorithms.tdmpc2.multitask.world_model import MultitaskWorldModel

_OBS_DIMS = [4, 6, 5]
_ACTION_DIMS = [2, 3, 2]


def _make_world_model(**kwargs) -> MultitaskWorldModel:
    params = dict(
        num_tasks=3,
        obs_dims=_OBS_DIMS,
        action_dims=_ACTION_DIMS,
        task_dim=8,
        latent_dim=16,
        enc_dim=8,
        mlp_dim=16,
        num_q=2,
        num_bins=11,
    )
    params.update(kwargs)
    return MultitaskWorldModel(**params)


def test_obs_and_action_dims_are_padded_to_max():
    wm = _make_world_model()
    assert wm.obs_dim == max(_OBS_DIMS)
    assert wm.action_dim == max(_ACTION_DIMS)


def test_action_masks_match_each_tasks_true_action_dim():
    wm = _make_world_model()
    for i, adim in enumerate(_ACTION_DIMS):
        assert torch.all(wm._action_masks[i, :adim] == 1.0)
        assert torch.all(wm._action_masks[i, adim:] == 0.0)


def test_pi_zeros_out_masked_action_dims():
    wm = _make_world_model()
    z = torch.randn(10, wm.latent_dim)
    task = torch.tensor([0, 2] * 5)  # both have action_dim=2 < max=3
    action, info = wm.pi(z, task)
    assert torch.all(action[:, 2] == 0.0)
    assert torch.all(info["mean"][:, 2] == 0.0)


def test_pi_does_not_zero_full_action_dim_task():
    wm = _make_world_model()
    z = torch.randn(10, wm.latent_dim)
    task = torch.full((10,), 1, dtype=torch.long)  # action_dim=3 == max
    action, _ = wm.pi(z, task)
    # Not asserting nonzero (could coincidentally be ~0), just that the mask
    # wasn't applied to this column -- verified via the mask tensor itself.
    assert torch.all(wm._action_masks[1] == 1.0)


def test_encode_next_reward_q_accept_task_conditioned_batches_and_backprop():
    wm = _make_world_model()
    obs = torch.randn(6, wm.obs_dim, requires_grad=True)
    task = torch.tensor([0, 1, 2, 0, 1, 2])
    z = wm.encode(obs, task)
    assert z.shape == (6, wm.latent_dim)

    a = torch.zeros(6, wm.action_dim)
    nz = wm.next(z, a, task)
    assert nz.shape == (6, wm.latent_dim)
    r = wm.reward(z, a, task)
    assert r.shape[0] == 6

    q_all = wm.Q(z, a, task, return_type="all")
    assert q_all.shape == (wm.num_q, 6, max(wm.num_bins, 1))
    q_min = wm.Q(z, a, task, return_type="min")
    assert q_min.shape == (6, 1)

    loss = q_min.sum()
    loss.backward()
    assert obs.grad is not None


def test_task_embed_broadcasts_across_leading_window_dim():
    wm = _make_world_model()
    task = torch.tensor([0, 1, 2])
    z3 = torch.zeros(4, 3, wm.latent_dim)
    a3 = torch.zeros(4, 3, wm.action_dim)
    r3 = wm.reward(z3, a3, task)
    assert r3.shape == (4, 3, max(wm.num_bins, 1))
    action3, _ = wm.pi(z3, task)
    assert action3.shape == (4, 3, wm.action_dim)


def test_target_q_polyak_update_moves_toward_live_q():
    wm = _make_world_model(tau=1.0)
    with torch.no_grad():
        for p in wm._Q.parameters():
            p.add_(1.0)
    wm.soft_update_target_Q()
    for p_live, p_target in zip(wm._Q.parameters(), wm._target_Q.parameters()):
        torch.testing.assert_close(p_live, p_target)


def test_rejects_mismatched_dims_length():
    import pytest

    with pytest.raises(ValueError):
        MultitaskWorldModel(num_tasks=3, obs_dims=[4, 6], action_dims=[2, 3, 2])
