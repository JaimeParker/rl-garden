from __future__ import annotations

import math

import pytest
import torch

from rl_garden.networks import DiffusionMLP


def test_output_shape_plain():
    net = DiffusionMLP(action_dim=3, horizon_steps=4, cond_dim=11, mlp_dims=(64, 64))
    x = torch.randn(5, 4, 3)
    t = torch.randint(0, 20, (5,))
    cond = {"state": torch.randn(5, 1, 11)}
    out = net(x, t, cond)
    assert out.shape == (5, 4, 3)


def test_output_shape_residual():
    net = DiffusionMLP(
        action_dim=3,
        horizon_steps=4,
        cond_dim=11,
        mlp_dims=(64, 64, 64),
        residual_style=True,
        activation_fn="relu",
    )
    x = torch.randn(5, 4, 3)
    t = torch.randint(0, 20, (5,))
    cond = {"state": torch.randn(5, 1, 11)}
    out = net(x, t, cond)
    assert out.shape == (5, 4, 3)


def test_residual_style_requires_odd_mlp_dims_count():
    with pytest.raises(ValueError, match="residual_style"):
        DiffusionMLP(
            action_dim=3, horizon_steps=4, cond_dim=11, mlp_dims=(64, 64), residual_style=True
        )


def test_sinusoidal_time_embedding_matches_reference_formula():
    from rl_garden.networks.diffusion_mlp import _SinusoidalPosEmb

    dim = 16
    emb = _SinusoidalPosEmb(dim)
    t = torch.tensor([0.0, 5.0, 19.0])
    out = emb(t)

    half_dim = dim // 2
    scale = math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
    expected = torch.cat(
        [torch.sin(t[:, None] * freqs[None, :]), torch.cos(t[:, None] * freqs[None, :])], dim=-1
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_gradients_flow_to_all_parameters():
    net = DiffusionMLP(
        action_dim=2,
        horizon_steps=2,
        cond_dim=4,
        mlp_dims=(32, 32, 32),
        residual_style=True,
    )
    x = torch.randn(3, 2, 2)
    t = torch.randint(0, 10, (3,))
    cond = {"state": torch.randn(3, 1, 4)}
    out = net(x, t, cond)
    out.sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
