"""PPO loss terms, update diagnostics and checkpoint metadata shared by the
PPO family: ``PPO`` and its subclasses, ``ChainPPOCore`` (``DPPO``,
``FlowPPO``), ``BPPOCore`` and ``UniO4`` (the clipped surrogate only).

Every function is the exact op sequence its callers ran inline before it was
shared, so moving a caller onto it does not change its numerics.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Union

import torch
import torch.nn.functional as F

#: A clip coefficient: one scalar, or one value per row of the minibatch
#: (``DPPO``'s per-denoising-step schedule).
ClipCoef = Union[float, torch.Tensor]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over the rows where ``mask`` is true; 0 when none is.
    ``count`` is ``mask.sum()`` (passed in so callers reuse it).
    ``torch.where`` (not a multiply) keeps non-finite masked rows out. The
    reduction ``PPO._aux_loss`` applies to a per-row loss over decision rows."""
    return torch.where(mask, x, 0.0).sum() / count.clamp_min(1)


def ppo_clip_policy_loss(
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    clip_coef: ClipCoef,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PPO's clipped surrogate objective: ``mean(max(-adv*ratio, -adv*clip(ratio)))``.

    ``clip_coef`` is a float or a tensor broadcasting to ``ratio`` (one
    coefficient per row). Shared by ``PPO._policy_loss``, ``ChainPPOCore``,
    ``BPPOCore`` and ``UniO4`` -- BPPO reuses only this formula, not
    ``_ppo_minibatch_update``, since it normalizes/omega-weights advantages
    in a different order and has no rollout buffer or value-clipping to hook
    into.

    ``mask`` (bool, same shape as ``advantages``) averages over the true rows
    only; masked rows contribute exactly zero (0 when no row is true).
    """
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    if mask is None:
        return torch.max(pg_loss1, pg_loss2).mean()
    return masked_mean(torch.max(pg_loss1, pg_loss2), mask, mask.sum())


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float | None,
) -> torch.Tensor:
    """The scalar PPO value loss over 1-D rows: ``0.5 * mse(values,
    returns)``; with a ``clip_coef``, ``values`` is also clipped to within
    ``clip_coef`` of the rollout's ``old_values`` and each row takes the
    larger of the two squared errors."""
    if clip_coef is None:
        return 0.5 * F.mse_loss(values, returns)
    v_loss_unclipped = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = (values_clipped - returns) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def approx_kl_clipfrac(
    logratio: torch.Tensor,
    ratio: torch.Tensor,
    clip_coef: ClipCoef,
    row_mean: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> tuple[torch.Tensor, float]:
    """``(approx_kl, clipfrac)`` of one minibatch, without gradients: the
    ``(ratio - 1) - logratio`` KL estimator (a detached scalar tensor) and
    the fraction of rows whose ratio left ``1 +- clip_coef`` (read to the
    host, one sync). ``row_mean`` reduces over the rows (a masked mean under
    a decision mask)."""
    with torch.no_grad():
        approx_kl = row_mean((ratio - 1.0) - logratio)
        clipfrac = row_mean(((ratio - 1.0).abs() > clip_coef).float()).item()
    return approx_kl, clipfrac


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """``1 - var(returns - values) / var(returns)`` over 1-D rows; NaN when
    the returns have no variance."""
    y_var = torch.var(returns)
    return float("nan") if y_var == 0 else float((1 - torch.var(returns - values) / y_var).item())


def encoder_checkpoint_metadata(algorithm: Any) -> dict[str, Any]:
    """The observation-encoder settings an algorithm records in its
    checkpoint metadata: ``encoder_sharing`` (and its origin),
    ``encoder_config``, ``obs_groups`` and ``critic_encoder_config``, the
    dataclasses as dicts. Reads those attributes of ``algorithm``."""
    return {
        "encoder_sharing": algorithm.encoder_sharing,
        "encoder_sharing_origin": algorithm.encoder_sharing_origin,
        "encoder_config": _asdict_or_none(algorithm.encoder_config),
        "obs_groups": _asdict_or_none(algorithm.obs_groups),
        "critic_encoder_config": _asdict_or_none(algorithm.critic_encoder_config),
    }


def _asdict_or_none(value: Any) -> dict[str, Any] | None:
    return dataclasses.asdict(value) if value is not None else None


__all__ = [
    "ClipCoef",
    "approx_kl_clipfrac",
    "clipped_value_loss",
    "encoder_checkpoint_metadata",
    "explained_variance",
    "masked_mean",
    "ppo_clip_policy_loss",
]
