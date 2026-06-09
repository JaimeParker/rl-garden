"""Offline Q-function contrast diagnostics.

The helpers here are intentionally algorithm-light: callers provide policy
objects that expose ``extract_features()``, ``predict()``, and ``q_values_all()``.
This keeps the diagnostic usable for IQL and SAC-family offline agents without
coupling it to their training losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class QContrastConfig:
    num_uniform_actions: int = 20
    percentiles: tuple[float, ...] = (10.0, 50.0, 90.0, 99.0)


def summary_stats(
    values: torch.Tensor,
    *,
    prefix: str,
    percentiles: Iterable[float] = (50.0, 90.0, 99.0),
) -> dict[str, float]:
    """Return finite scalar summary stats for a tensor."""
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"Cannot summarize empty tensor for {prefix!r}.")
    if not torch.isfinite(flat).all():
        raise ValueError(f"Non-finite values found while summarizing {prefix!r}.")

    out = {
        f"{prefix}/mean": float(flat.mean().item()),
        f"{prefix}/std": float(flat.std(unbiased=False).item()),
        f"{prefix}/min": float(flat.min().item()),
        f"{prefix}/max": float(flat.max().item()),
    }
    for p in percentiles:
        q = torch.quantile(flat, float(p) / 100.0)
        out[f"{prefix}/p{int(p):02d}"] = float(q.item())
    return out


def uniform_actions(
    *,
    batch_size: int,
    num_actions: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-state uniform actions with shape ``(B, K, action_dim)``."""
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}.")
    low = action_low.float()
    high = action_high.float()
    sample = torch.rand(
        batch_size,
        num_actions,
        *low.shape,
        device=low.device,
        generator=generator,
    )
    return low + (high - low) * sample


def compute_action_grad_norm(
    policy: Any,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    ensemble_reduce: str = "min",
) -> torch.Tensor:
    """Compute ``||d Q(s, a) / d a||_2`` for each action row."""
    if ensemble_reduce not in {"min", "mean"}:
        raise ValueError(f"Unknown ensemble_reduce={ensemble_reduce!r}.")

    probe_actions = actions.detach().clone().requires_grad_(True)
    features = policy.extract_features(obs, stop_gradient=False)
    q_all = policy.q_values_all(features, probe_actions, target=False)
    if ensemble_reduce == "min":
        q = q_all.min(dim=0).values
    else:
        q = q_all.mean(dim=0)
    grad = torch.autograd.grad(q.sum(), probe_actions, retain_graph=False)[0]
    return grad.norm(dim=-1)


def compute_q_contrast_metrics(
    policy: Any,
    obs: torch.Tensor,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    config: QContrastConfig = QContrastConfig(),
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Compute offline Q contrast metrics on a batch of Box observations."""
    if not isinstance(obs, torch.Tensor):
        raise TypeError("Q contrast diagnostics currently require Tensor/Box observations.")
    if obs.ndim < 2:
        raise ValueError(f"Expected batched observations, got shape={tuple(obs.shape)}.")

    batch_size = obs.shape[0]
    actions_uniform = uniform_actions(
        batch_size=batch_size,
        num_actions=config.num_uniform_actions,
        action_low=action_low.to(obs.device),
        action_high=action_high.to(obs.device),
        generator=generator,
    )

    metrics: dict[str, float] = {}
    with torch.no_grad():
        policy_actions = policy.predict(obs, deterministic=True)
        features = policy.extract_features(obs, stop_gradient=False)
        q_policy_all = policy.q_values_all(features, policy_actions, target=False)

        flat_uniform = actions_uniform.reshape(batch_size * config.num_uniform_actions, -1)
        obs_repeated = obs.repeat_interleave(config.num_uniform_actions, dim=0)
        features_uniform = policy.extract_features(obs_repeated, stop_gradient=False)
        q_uniform_all = policy.q_values_all(features_uniform, flat_uniform, target=False)
        q_uniform_all = q_uniform_all.reshape(
            q_uniform_all.shape[0], batch_size, config.num_uniform_actions
        )

    q_policy_min = q_policy_all.min(dim=0).values
    q_policy_mean_ensemble = q_policy_all.mean(dim=0)
    q_uniform_min = q_uniform_all.min(dim=0).values
    q_uniform_mean_ensemble = q_uniform_all.mean(dim=0)

    for name, q_uniform, q_policy in (
        ("min", q_uniform_min, q_policy_min),
        ("mean_ensemble", q_uniform_mean_ensemble, q_policy_mean_ensemble),
    ):
        var_uniform = q_uniform.var(dim=1, unbiased=False)
        range_uniform = q_uniform.max(dim=1).values - q_uniform.min(dim=1).values
        uniform_mean = q_uniform.mean(dim=1)
        policy_uniform_gap = q_policy - uniform_mean
        percentiles = config.percentiles
        metrics.update(
            summary_stats(
                var_uniform,
                prefix=f"q_contrast/{name}/var_uniform",
                percentiles=percentiles,
            )
        )
        metrics.update(
            summary_stats(
                range_uniform,
                prefix=f"q_contrast/{name}/range_uniform",
                percentiles=percentiles,
            )
        )
        metrics.update(
            summary_stats(
                policy_uniform_gap,
                prefix=f"q_contrast/{name}/policy_uniform_gap",
                percentiles=percentiles,
            )
        )
        metrics.update(
            summary_stats(
                q_policy,
                prefix=f"q_contrast/{name}/q_policy",
                percentiles=percentiles,
            )
        )
        metrics.update(
            summary_stats(
                uniform_mean,
                prefix=f"q_contrast/{name}/q_uniform_mean",
                percentiles=percentiles,
            )
        )

    grad_policy_min = compute_action_grad_norm(
        policy, obs, policy_actions, ensemble_reduce="min"
    )
    grad_policy_mean = compute_action_grad_norm(
        policy, obs, policy_actions, ensemble_reduce="mean"
    )
    flat_uniform_grad_actions = actions_uniform.reshape(
        batch_size * config.num_uniform_actions, -1
    )
    obs_repeated = obs.repeat_interleave(config.num_uniform_actions, dim=0)
    grad_uniform_min = compute_action_grad_norm(
        policy, obs_repeated, flat_uniform_grad_actions, ensemble_reduce="min"
    ).reshape(batch_size, config.num_uniform_actions)
    grad_uniform_mean = compute_action_grad_norm(
        policy, obs_repeated, flat_uniform_grad_actions, ensemble_reduce="mean"
    ).reshape(batch_size, config.num_uniform_actions)

    for name, grad_policy, grad_uniform in (
        ("min", grad_policy_min, grad_uniform_min.mean(dim=1)),
        ("mean_ensemble", grad_policy_mean, grad_uniform_mean.mean(dim=1)),
    ):
        metrics.update(
            summary_stats(
                grad_policy,
                prefix=f"q_contrast/{name}/grad_norm_policy",
                percentiles=config.percentiles,
            )
        )
        metrics.update(
            summary_stats(
                grad_uniform,
                prefix=f"q_contrast/{name}/grad_norm_uniform",
                percentiles=config.percentiles,
            )
        )

    return metrics
