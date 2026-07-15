"""Optional obs mean/std normalization for Box-observation policies.

``ObsNormalizingMixin`` stores normalization statistics as policy buffers so
they round-trip through ``state_dict()`` (checkpoint save/load) automatically,
and applies them inside ``extract_features()`` so every consumer of features
(training, eval, and Off2On online rollout) sees normalized observations
through a single entry point. Statistics are fit once from the offline
dataset and frozen afterward (Off2On online rollout does not update them),
matching CORL's ``compute_mean_std``/``normalize_states`` convention for
TD3-BC/AWAC.
"""
from __future__ import annotations

import torch


class ObsNormalizingMixin:
    """Mixin for Box-observation policies that normalize obs by mean/std."""

    def _register_obs_normalizer(self, obs_dim: int) -> None:
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def fit_obs_normalizer(self, obs: torch.Tensor, eps: float = 1e-3) -> None:
        """Fit ``obs_mean``/``obs_std`` from a ``(N, obs_dim)`` tensor of
        observations. Intended to be called once, from the offline dataset,
        before training starts."""
        mean = obs.mean(dim=0).to(self.obs_mean.device, self.obs_mean.dtype)
        std = obs.std(dim=0).to(self.obs_std.device, self.obs_std.dtype) + eps
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(std)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std
