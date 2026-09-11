from __future__ import annotations

import pytest
import torch

from rl_garden.networks import DiffusionUNet1D


def test_output_shape_default_backbone():
    net = DiffusionUNet1D(
        action_dim=3, horizon_steps=4, cond_dim=11, time_dim=16, down_dims=(8, 16)
    )
    x = torch.randn(5, 4, 3)
    t = torch.randint(0, 20, (5,))
    cond = {"state": torch.randn(5, 1, 11)}
    out = net(x, t, cond)
    assert out.shape == (5, 4, 3)


def test_output_shape_cond_predict_scale():
    net = DiffusionUNet1D(
        action_dim=2,
        horizon_steps=8,
        cond_dim=6,
        time_dim=16,
        down_dims=(8, 16, 32),
        cond_predict_scale=True,
    )
    x = torch.randn(3, 8, 2)
    t = torch.randint(0, 10, (3,))
    cond = {"state": torch.randn(3, 1, 6)}
    out = net(x, t, cond)
    assert out.shape == (3, 8, 2)


def test_horizon_steps_not_divisible_raises():
    with pytest.raises(ValueError, match="horizon_steps"):
        DiffusionUNet1D(
            action_dim=2, horizon_steps=3, cond_dim=6, time_dim=16, down_dims=(8, 16, 32)
        )


def test_gradients_flow_to_all_parameters():
    net = DiffusionUNet1D(
        action_dim=2, horizon_steps=4, cond_dim=5, time_dim=16, down_dims=(8, 16)
    )
    x = torch.randn(3, 4, 2)
    t = torch.randint(0, 10, (3,))
    cond = {"state": torch.randn(3, 1, 5)}
    out = net(x, t, cond)
    out.sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
