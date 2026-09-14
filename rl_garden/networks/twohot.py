"""Two-hot scalar<->distribution encoding, ported from
``3rd_party/tdmpc2/tdmpc2/common/math.py``.

Shared numeric primitive: TD-MPC2's reward/value heads and DreamerV3's
reward/critic heads both predict a discrete distribution over ``num_bins``
symlog-spaced bins instead of a scalar, trained with ``soft_ce`` against a
soft two-hot target, and decoded back to a scalar with ``two_hot_inv``.

``symexp_twohot_bins``/``twohot_logprob``/``twohot_mean`` below are
DreamerV3's *own* two-hot variant (r2dreamer ``distributions.py:67-129,242-
251``; plan section A) -- symexp-spaced bins fixed at construction (not
linearly spaced between a learned ``vmin``/``vmax`` like ``two_hot`` above),
and an exact float target rather than a symlog-clamped one. Kept as separate
functions rather than folded into ``two_hot``/``two_hot_inv`` above: the bin
construction, target-to-bin search, and mode decoding are all genuinely
different algorithms, not just different constants.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_garden.networks.symlog import symexp, symlog


def two_hot(
    x: torch.Tensor, num_bins: int, vmin: float, vmax: float, bin_size: float
) -> torch.Tensor:
    """Converts a batch of scalars to soft two-hot encoded targets."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(-1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Converts a batch of soft two-hot encoded vectors back to scalars."""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def soft_ce(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_bins: int,
    vmin: float,
    vmax: float,
    bin_size: float,
) -> torch.Tensor:
    """Cross-entropy loss between predicted bin logits and soft two-hot targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, num_bins, vmin, vmax, bin_size)
    return -(target * pred).sum(-1, keepdim=True)


def symexp_twohot_bins(num_bins: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """DreamerV3's symexp-spaced two-hot bin edges, ported verbatim from
    r2dreamer's ``symexp_twohot`` factory (``distributions.py:242-251`` --
    identical in the official JAX ``heads.py:132-144``): ``symexp`` of a
    linspace over ``[-20, 0]`` covering half the bins, mirrored to be
    symmetric about 0. Ascending 1-D tensor of length ``num_bins`` (e.g. 255
    for the reward/critic heads); ``bins[(num_bins - 1) // 2] == 0`` when
    ``num_bins`` is odd.
    """
    if num_bins % 2 == 1:
        half = symexp(torch.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1, device=device))
        return torch.cat([half, -half[:-1].flip(dims=(0,))], dim=0)
    half = symexp(torch.linspace(-20.0, 0.0, num_bins // 2, device=device))
    return torch.cat([half, -half.flip(dims=(0,))], dim=0)


def twohot_logprob(logits: torch.Tensor, bins: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-probability of ``target`` (``(..., 1)``, raw scale -- NOT
    symlog'd: the symexp-spaced ``bins`` already cover the wide dynamic
    range) under the two-hot distribution parameterized by ``logits``
    (``(..., num_bins)``). Ported verbatim from r2dreamer's
    ``TwoHot.log_prob`` (``distributions.py:100-129``) with its default
    identity ``squash``/``unsquash`` (``symexp_twohot`` never overrides
    them). Returns ``(...,)`` (the trailing bin dim is reduced away).
    """
    target = target.squeeze(-1)
    target_detached = target.detach()
    below = (bins <= target_detached.unsqueeze(-1)).sum(dim=-1) - 1
    above = bins.shape[-1] - (bins > target_detached.unsqueeze(-1)).sum(dim=-1)
    below = torch.clamp(below, 0, bins.shape[-1] - 1)
    above = torch.clamp(above, 0, bins.shape[-1] - 1)
    equal = below == above
    dist_to_below = torch.where(
        equal, torch.ones_like(target_detached), (bins[below] - target_detached).abs()
    )
    dist_to_above = torch.where(
        equal, torch.ones_like(target_detached), (bins[above] - target_detached).abs()
    )
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    oh_below = F.one_hot(below, num_classes=bins.shape[-1]).to(logits.dtype)
    oh_above = F.one_hot(above, num_classes=bins.shape[-1]).to(logits.dtype)
    mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
    log_pred = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return (mixed_target * log_pred).sum(dim=-1)


def twohot_mean(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decodes ``logits`` (``(..., num_bins)``) to a scalar (``(..., 1)``) via
    the symmetric flip-pair weighted average, ported verbatim from
    r2dreamer's ``TwoHot.mode`` (``distributions.py:78-98``) -- deliberately
    NOT a plain ``(softmax(logits) * bins).sum(-1)`` dot product: ``bins``
    span roughly ``+/- expm1(20) ~= 4.85e8``, and this pairwise flip-and-add
    (small-magnitude bins first) keeps the weighted sum numerically
    symmetric in float32, where a naive dot product degrades.
    """
    probs = torch.softmax(logits, dim=-1)
    n = logits.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1, p2, p3 = probs[..., :m], probs[..., m : m + 1], probs[..., m + 1 :]
        b1, b2, b3 = bins[:m], bins[m : m + 1], bins[m + 1 :]
        return (p2 * b2).sum(dim=-1, keepdim=True) + (
            (p1 * b1).flip(dims=(-1,)) + (p3 * b3)
        ).sum(dim=-1, keepdim=True)
    p1, p2 = probs[..., : n // 2], probs[..., n // 2 :]
    b1, b2 = bins[: n // 2], bins[n // 2 :]
    return ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
