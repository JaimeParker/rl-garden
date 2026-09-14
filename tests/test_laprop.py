"""Unit tests for ``rl_garden.common.optim.laprop`` (``LaProp``, ``linear_warmup``)
and ``rl_garden.common.optim.agc`` (``clip_grad_agc_``), DreamerV3's optimizer
(plan model-based-base Part 2 / decision 1).
"""
from __future__ import annotations

import math

import torch

from rl_garden.common.optim.agc import clip_grad_agc_
from rl_garden.common.optim.laprop import LaProp, linear_warmup


def test_laprop_first_step_moves_by_lr_times_sign_of_gradient():
    # LaProp's defining property (see laprop.py's module docstring): with all
    # optimizer state at zero, the very first step's exp_avg_sq after the
    # update is exactly (1-beta2)*grad**2, so denom == |grad| exactly (its
    # own bias-correction term is also (1-beta2), which cancels); step_size
    # == 1/(1-beta1) exactly cancels the (1-beta1) baked into exp_avg. The
    # parameter therefore moves by exactly lr*sign(grad) on step 1,
    # regardless of beta1/beta2/the gradient's magnitude.
    lr = 4e-5
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([3.7])  # arbitrary nonzero magnitude
    opt = LaProp([param], lr=lr, betas=(0.9, 0.999), eps=1e-20)

    before = param.detach().clone()
    opt.step()

    expected = before - lr  # sign(grad) = +1
    torch.testing.assert_close(param.detach(), expected)


def test_laprop_first_step_moves_by_lr_times_sign_of_negative_gradient():
    lr = 4e-5
    param = torch.nn.Parameter(torch.tensor([-2.0]))
    param.grad = torch.tensor([-0.01])
    opt = LaProp([param], lr=lr, betas=(0.9, 0.999), eps=1e-20)

    before = param.detach().clone()
    opt.step()

    expected = before + lr  # sign(grad) = -1
    torch.testing.assert_close(param.detach(), expected)


def test_linear_warmup_ramps_then_holds():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = LaProp([param], lr=1e-3)
    sched = linear_warmup(opt, warmup_steps=10)

    lrs = []
    for _ in range(12):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    assert math.isclose(lrs[0], 1e-3 * (1 / 10))
    assert math.isclose(lrs[5], 1e-3 * (6 / 10))
    assert math.isclose(lrs[9], 1e-3 * 1.0)
    assert math.isclose(lrs[11], 1e-3 * 1.0)  # holds at peak past warmup


def test_agc_leaves_small_gradients_unscaled():
    # grad_norm / (clip * max(pmin, param_norm)) < 1 -> scale factor is
    # clamped to 1, gradient untouched.
    param = torch.nn.Parameter(torch.full((4,), 10.0))  # large param norm
    param.grad = torch.full((4,), 1e-4)  # tiny gradient
    original_grad = param.grad.clone()

    clip_grad_agc_([param], clip=0.3, pmin=1e-3)

    torch.testing.assert_close(param.grad, original_grad)


def test_agc_scales_down_large_gradients():
    param = torch.nn.Parameter(torch.full((4,), 1e-4))  # tiny param norm
    param.grad = torch.full((4,), 10.0)  # huge gradient

    clip_grad_agc_([param], clip=0.3, pmin=1e-3)

    pnorm = torch.norm(torch.full((4,), 1e-4), p=2)
    upper = 0.3 * torch.maximum(torch.tensor(1e-3), pnorm)
    gnorm = torch.norm(torch.full((4,), 10.0), p=2)
    expected_scale = 1.0 / torch.maximum(torch.tensor(1.0), gnorm / upper)
    expected_grad = torch.full((4,), 10.0) * expected_scale

    torch.testing.assert_close(param.grad, expected_grad)
    assert torch.norm(param.grad, p=2).item() < torch.norm(torch.full((4,), 10.0), p=2).item()
