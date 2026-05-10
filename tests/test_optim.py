"""Unit tests for rl_garden.common.optim."""
from __future__ import annotations

import math

import pytest
import torch

from rl_garden.common.optim import make_lr_scheduler, make_optimizer


def _dummy_params():
    return [torch.nn.Parameter(torch.zeros(4))]


def test_make_optimizer_default_is_adam():
    opt = make_optimizer(_dummy_params(), lr=1e-3)
    assert isinstance(opt, torch.optim.Adam)
    assert not isinstance(opt, torch.optim.AdamW)


def test_make_optimizer_promotes_to_adamw_when_weight_decay_positive():
    opt = make_optimizer(_dummy_params(), lr=1e-3, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == 1e-4


def test_make_optimizer_explicit_adamw_with_zero_wd():
    opt = make_optimizer(_dummy_params(), lr=1e-3, use_adamw=True)
    assert isinstance(opt, torch.optim.AdamW)


def test_make_lr_scheduler_constant_returns_none():
    opt = make_optimizer(_dummy_params(), lr=1e-3)
    sched = make_lr_scheduler(opt, schedule_type="constant")
    assert sched is None


def test_linear_warmup_ramps_to_full_lr():
    opt = make_optimizer(_dummy_params(), lr=1.0)
    sched = make_lr_scheduler(
        opt, schedule_type="linear_warmup", warmup_steps=10
    )
    assert sched is not None
    # Initial step (step=0) -> 0% of base lr
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0)
    sched.step()  # step 1 -> 0.1
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    for _ in range(9):
        sched.step()  # eventually step 10 -> 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)
    # Past warmup, stays flat
    for _ in range(20):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_warmup_cosine_shape():
    opt = make_optimizer(_dummy_params(), lr=1.0)
    sched = make_lr_scheduler(
        opt,
        schedule_type="warmup_cosine",
        warmup_steps=5,
        decay_steps=10,
        min_lr_ratio=0.1,
    )
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(20):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # Warmup: lrs[0..5] should be increasing 0,0.2,0.4,0.6,0.8,1.0
    for i in range(6):
        assert lrs[i] == pytest.approx(i / 5.0)
    # Cosine decay: at step 5 (start of decay) lr=1.0; at step 15 (end of decay) lr=min_lr_ratio
    # cos(0)=1 -> lr = 0.1 + 0.9*1.0 = 1.0
    # cos(pi)= -1 -> lr = 0.1 + 0.9*0 = 0.1
    assert lrs[5] == pytest.approx(1.0)
    assert lrs[15] == pytest.approx(0.1, abs=1e-6)
    # After decay completes, lr stays at min
    assert lrs[20] == pytest.approx(0.1, abs=1e-6)


def test_warmup_cosine_requires_decay_steps():
    opt = make_optimizer(_dummy_params(), lr=1.0)
    with pytest.raises(ValueError, match="warmup_cosine requires decay_steps"):
        make_lr_scheduler(opt, schedule_type="warmup_cosine", warmup_steps=5, decay_steps=0)


def test_invalid_min_lr_ratio_raises():
    opt = make_optimizer(_dummy_params(), lr=1.0)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        make_lr_scheduler(
            opt, schedule_type="warmup_cosine", warmup_steps=5,
            decay_steps=10, min_lr_ratio=1.5,
        )


def test_unknown_schedule_raises():
    opt = make_optimizer(_dummy_params(), lr=1.0)
    with pytest.raises(ValueError, match="Unknown schedule_type"):
        make_lr_scheduler(opt, schedule_type="bogus")  # type: ignore[arg-type]
