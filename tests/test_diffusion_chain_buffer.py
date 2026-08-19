from __future__ import annotations

import pytest
import torch

from rl_garden.buffers.diffusion_chain_buffer import DiffusionChainBuffer


def test_add_fills_in_order_and_tracks_full():
    buf = DiffusionChainBuffer(
        num_steps=3, num_envs=2, ft_denoising_steps=4, horizon_steps=2, action_dim=3, device="cpu"
    )
    assert not buf.full
    for i in range(3):
        chain = torch.full((2, 5, 2, 3), float(i))
        log_probs = torch.full((2, 4, 2, 3), float(-i))
        buf.add(chain, log_probs)
    assert buf.full
    assert buf.chains.shape == (3, 2, 5, 2, 3)
    assert buf.old_log_probs.shape == (3, 2, 4, 2, 3)
    assert torch.equal(buf.chains[1], torch.full((2, 5, 2, 3), 1.0))
    assert torch.equal(buf.old_log_probs[2], torch.full((2, 4, 2, 3), -2.0))


def test_add_past_capacity_raises():
    buf = DiffusionChainBuffer(
        num_steps=1, num_envs=1, ft_denoising_steps=2, horizon_steps=1, action_dim=1, device="cpu"
    )
    buf.add(torch.zeros(1, 3, 1, 1), torch.zeros(1, 2, 1, 1))
    with pytest.raises(RuntimeError):
        buf.add(torch.zeros(1, 3, 1, 1), torch.zeros(1, 2, 1, 1))


def test_reset_allows_refill():
    buf = DiffusionChainBuffer(
        num_steps=1, num_envs=1, ft_denoising_steps=2, horizon_steps=1, action_dim=1, device="cpu"
    )
    buf.add(torch.zeros(1, 3, 1, 1), torch.zeros(1, 2, 1, 1))
    buf.reset()
    assert not buf.full
    assert buf.pos == 0
    buf.add(torch.ones(1, 3, 1, 1), torch.ones(1, 2, 1, 1))
    assert buf.full
