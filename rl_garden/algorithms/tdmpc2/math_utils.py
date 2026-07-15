"""Pure tensor math ported from ``3rd_party/tdmpc2/tdmpc2/common/math.py``.

Multitask-specific branches (task embeddings, per-task action masks) are
dropped -- this port is single-task only, see
``rl_garden/algorithms/tdmpc2/agent.py``'s module docstring.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm (Dreamer-v3 style)."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of ``symlog``."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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


def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std_: torch.Tensor) -> torch.Tensor:
    """Gaussian log-probability of a reparameterized sample."""
    residual = -0.5 * eps.pow(2) - log_std_
    log_prob = residual - 0.9189385175704956  # 0.5 * log(2*pi)
    return log_prob.sum(-1, keepdim=True)


def squash(
    mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply tanh squashing and correct the log-probability accordingly."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0) -> torch.Tensor:
    """Sample an index from the Gumbel-Softmax distribution over probabilities ``p``."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)


def termination_statistics(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-9
) -> dict[str, torch.Tensor]:
    """Precision/recall/F1 diagnostics for the termination classifier."""
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    rate = target.sum() / len(target)
    tp = ((pred > 0.5) & (target == 1)).sum()
    fn = ((pred <= 0.5) & (target == 1)).sum()
    fp = ((pred > 0.5) & (target == 0)).sum()
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return {"termination_rate": rate, "termination_f1": f1}
