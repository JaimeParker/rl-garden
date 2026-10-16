"""Action masking for multi-embodiment policies whose action head is
zero-padded to a shared ``padded_action_dim`` (NCRL).

``masked_bounded_normal(mean, std, mask)`` multiplies the mean, every sample
and the per-dim entropy/log-prob by ``mask`` so padded dims are exactly zero
and contribute nothing to the objective. This also fixes the official NCRL
``sampe`` typo (``nets.py:374``), where samples were left unmasked
(deviation from the official code).
"""
from __future__ import annotations

import torch


class MaskedBoundedNormal:
    """Diagonal Normal over ``(..., A)`` whose padded dims (``mask == 0``) are
    pinned to zero. ``log_prob``/``entropy`` sum over the unmasked dims only."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor) -> None:
        self.mask = mask.to(mean.dtype)
        self._normal = torch.distributions.Normal(mean, std)

    @property
    def mean(self) -> torch.Tensor:
        return self._normal.mean * self.mask

    @property
    def mode(self) -> torch.Tensor:
        return self.mean

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._normal.rsample(sample_shape) * self.mask

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._normal.sample(sample_shape) * self.mask

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return (self._normal.log_prob(value) * self.mask).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return (self._normal.entropy() * self.mask).sum(dim=-1)


class MaskedTruncatedNormal(MaskedBoundedNormal):
    """NCRL's ``TruncatedNormal`` (``3rd_party/ncrl/src/dreamer/nets.py:29-53``)
    with the same masking: ``rsample``/``sample`` clamp ``mean + std * eps`` to
    ``[low + eps, high - eps]`` with a straight-through gradient (forward value
    clamped, gradient of the unclamped sample), then zero the padded dims.
    ``log_prob``/``entropy`` are the masked Normal's; the entropy is analytic,
    whose expectation and gradient equal the official single-sample
    ``-log_prob(sample)`` away from the clamp."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        mask: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(mean, std, mask)
        self.low, self.high, self.eps = low, high, eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._clamp(self._normal.rsample(sample_shape)) * self.mask

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)


def masked_bounded_normal(mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor) -> MaskedBoundedNormal:
    """``mean``/``std``: ``(..., A)`` (``mean`` already bounded, e.g. the
    ``tanh(mean)`` of ``rl_garden.networks.dreamer_nets.bounded_normal``).
    ``mask``: ``(..., A)`` or ``(A,)``, 1 for real action dims, 0 for padding."""
    return MaskedBoundedNormal(mean, std, mask)


def action_mask(action_dim: int, padded_action_dim: int, device: torch.device | str | None = None) -> torch.Tensor:
    """``(padded_action_dim,)`` float mask with the first ``action_dim`` dims set."""
    if not 0 < action_dim <= padded_action_dim:
        raise ValueError(f"need 0 < action_dim <= padded_action_dim, got {action_dim}, {padded_action_dim}.")
    mask = torch.zeros(padded_action_dim, device=device)
    mask[:action_dim] = 1.0
    return mask
