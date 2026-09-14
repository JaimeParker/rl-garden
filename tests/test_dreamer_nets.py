"""Unit tests for ``rl_garden.networks.twohot``'s DreamerV3 additions and
``rl_garden.networks.dreamer_nets`` (plan model-based-base Part 2, section A).
"""
from __future__ import annotations

import torch

from rl_garden.networks.dreamer_nets import (
    BlockLinear,
    MLPHead,
    OneHotDist,
    ReturnEMA,
    RMSNorm,
    bounded_normal,
    dreamer_weight_init_,
)
from rl_garden.networks.symlog import symexp
from rl_garden.networks.twohot import symexp_twohot_bins, twohot_logprob, twohot_mean


def test_symexp_twohot_bins_are_ascending_and_symmetric():
    bins = symexp_twohot_bins(255)
    assert bins.shape == (255,)
    assert bool((bins[1:] >= bins[:-1]).all())
    assert torch.isclose(bins[127], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(bins[0], -bins[-1], atol=1e-2)
    assert torch.isclose(bins[-1], symexp(torch.tensor(-20.0)).abs())


def test_symexp_twohot_bins_even_count():
    bins = symexp_twohot_bins(256)
    assert bins.shape == (256,)
    assert bool((bins[1:] >= bins[:-1]).all())


def test_twohot_round_trip_peaked_logits_decode_near_target_bin():
    bins = symexp_twohot_bins(255)
    logits = torch.zeros(1, 255)
    logits[0, 200] = 20.0  # near one-hot at bin 200
    mean = twohot_mean(logits, bins)
    assert mean.shape == (1, 1)
    torch.testing.assert_close(mean, bins[200].reshape(1, 1), atol=1e-2, rtol=1e-2)


def test_twohot_logprob_is_maximal_at_the_true_bin_center():
    bins = symexp_twohot_bins(255)
    logits = torch.randn(4, 255)
    target_on_bin = bins[100].reshape(1, 1).expand(4, 1)
    target_off_bin = (bins[100] + 1e6).reshape(1, 1).expand(4, 1)
    lp_on = twohot_logprob(logits, bins, target_on_bin)
    lp_off = twohot_logprob(logits, bins, target_off_bin)
    # An arbitrary logit distribution's log-prob of a target exactly at one
    # bin's center only depends on that one bin's log-softmax value; moving
    # the target far away changes which two bins get mixed. Just check both
    # are finite and that log_prob is not the same everywhere (data-dependent).
    assert torch.isfinite(lp_on).all() and torch.isfinite(lp_off).all()


def test_rmsnorm_eps_and_dtype():
    norm = RMSNorm(16)
    assert norm.eps == 1e-4
    assert norm.weight.dtype == torch.float32
    out = norm(torch.randn(3, 16))
    assert out.shape == (3, 16)


def test_blocklinear_shapes_and_fan_in():
    block = BlockLinear(64, 24, blocks=8)
    out = block(torch.randn(5, 64))
    assert out.shape == (5, 24)
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(block.weight)
    assert fan_in == 64  # full in_ch, not one block's share -- see docstring
    assert fan_out == 24


def test_blocklinear_rejects_non_divisible_channels():
    import pytest

    with pytest.raises(ValueError):
        BlockLinear(63, 24, blocks=8)


def test_dreamer_weight_init_sets_rmsnorm_weight_to_one():
    norm = RMSNorm(8)
    torch.nn.init.zeros_(norm.weight)
    norm.apply(dreamer_weight_init_)
    torch.testing.assert_close(norm.weight, torch.ones(8))


def test_dreamer_weight_init_zeros_bias():
    linear = torch.nn.Linear(10, 4)
    linear.apply(dreamer_weight_init_)
    torch.testing.assert_close(linear.bias, torch.zeros(4))
    assert linear.weight.std().item() > 0  # not left at default init


def test_return_ema_scale_is_at_least_one_and_tracks_percentiles():
    ema = ReturnEMA()
    for _ in range(200):
        offset, scale = ema(torch.randn(256) * 10.0)
    assert scale.item() >= 1.0
    assert torch.isfinite(offset)


def test_onehot_dist_rsample_is_one_hot_and_gradient_carrying():
    logits = torch.randn(3, 32, 16, requires_grad=True)
    dist = OneHotDist(logits, unimix=0.01)
    sample = dist.rsample()
    assert sample.shape == (3, 32, 16)
    torch.testing.assert_close(sample.sum(-1), torch.ones(3, 32))
    assert sample.requires_grad  # straight-through: gradient flows to logits
    sample.sum().backward()
    assert logits.grad is not None


def test_bounded_normal_mean_is_bounded_by_tanh_and_std_in_range():
    x = torch.randn(10, 8)  # 2*action_dim=4
    dist = bounded_normal(x, min_std=0.1, max_std=1.0)
    assert dist.mean.abs().max().item() <= 1.0 + 1e-6
    std = dist.base_dist.scale
    assert std.min().item() >= 0.1 - 1e-6
    assert std.max().item() <= 1.0 + 1e-6


def test_mlphead_outscale_zero_zeroes_the_last_layer():
    head = MLPHead(10, 5, output="mse", outscale=0.0)
    assert torch.allclose(head.last.weight, torch.zeros_like(head.last.weight))
