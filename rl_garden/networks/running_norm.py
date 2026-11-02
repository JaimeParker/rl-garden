"""``RunningNorm``: running mean/variance normalizer stored as buffers (so the
statistics are checkpointed with the owning module), merged with the exact
parallel-variance formula."""
from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Per-component running mean/var over the trailing ``shape`` dims.

    ``epsilon`` is the initial count (so the first batch dominates the
    ``mean=0, var=1`` prior); ``eps`` floors the denominator in ``forward`` /
    ``denormalize`` (``sqrt(var + eps**2)``). Buffers ``mean``, ``var``,
    ``count`` match PWM's return normalizer. Not
    ``rl_garden.common.reward_normalizer.RunningMeanStd``: that one is
    ``torch.compile``d and starts from count 0.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4, eps: float = 1e-8) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.eps = eps
        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("var", torch.ones(self.shape))
        self.register_buffer("count", torch.tensor(float(epsilon)))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """``x``: ``(..., *shape)``; leading dims are flattened into the sample axis."""
        x = x.reshape(-1, *self.shape).to(self.mean.dtype)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        return self.var.sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * torch.sqrt(self.var + self.eps**2) + self.mean
