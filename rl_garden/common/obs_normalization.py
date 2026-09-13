"""Optional obs mean/std normalization, applied by whichever policy owns an
``ObsNormalizingMixin``.

``ObsNormalizingMixin`` stores normalization statistics as policy buffers so
they round-trip through ``state_dict()`` (checkpoint save/load)
automatically. Statistics are fit once from the offline dataset and frozen
afterward (Off2On online rollout does not update them), matching CORL's
``compute_mean_std``/``normalize_states`` convention.

WHERE the mixin's ``_normalize_obs`` is applied is each caller's own choice,
not fixed by this module: ``TD3BCPolicy``/``BCQPolicy``/``PLASPolicy``
normalize the raw state entries (``schema.state_keys`` -- ``"state"`` plus
any ``state_<name>`` keys, concatenated/normalized/split back together)
inside ``extract_features()`` *before* the features extractor runs -- CORL's
own semantics, and the only choice that is meaningful for a Dict+image
schema (the extractor's own image branch is never touched). ``AWACPolicy``
instead normalizes the extractor's
output features (state-only by convention there, so numerically equivalent
to normalizing the raw state); since ``encoder_sharing="separate"`` gives it
a critic_extractor whose output dim can differ from the actor extractor's,
``AWACPolicy`` registers a second, ``suffix="critic"`` buffer pair only in
that case (the default shared case keeps exactly one buffer pair, unchanged).

Each buffer pair is named ``obs_mean<suffix>``/``obs_std<suffix>`` --
``suffix=""`` (the default) reproduces the original unsuffixed
``obs_mean``/``obs_std`` names every existing caller already relies on.
"""
from __future__ import annotations

import torch


class ObsNormalizingMixin:
    """Mixin giving a policy mean/std buffers plus fit/normalize helpers;
    see the module docstring for where each caller applies them."""

    def _register_obs_normalizer(self, obs_dim: int, *, suffix: str = "") -> None:
        self.register_buffer(f"obs_mean{suffix}", torch.zeros(obs_dim))
        self.register_buffer(f"obs_std{suffix}", torch.ones(obs_dim))

    def fit_obs_normalizer(self, obs: torch.Tensor, eps: float = 1e-3, *, suffix: str = "") -> None:
        """Fit ``obs_mean<suffix>``/``obs_std<suffix>`` from a ``(N, obs_dim)``
        tensor of observations. Intended to be called once, from the offline
        dataset, before training starts."""
        obs_mean = getattr(self, f"obs_mean{suffix}")
        obs_std = getattr(self, f"obs_std{suffix}")
        mean = obs.mean(dim=0).to(obs_mean.device, obs_mean.dtype)
        std = obs.std(dim=0).to(obs_std.device, obs_std.dtype) + eps
        obs_mean.copy_(mean)
        obs_std.copy_(std)

    def _normalize_obs(self, obs: torch.Tensor, *, suffix: str = "") -> torch.Tensor:
        obs_mean = getattr(self, f"obs_mean{suffix}")
        obs_std = getattr(self, f"obs_std{suffix}")
        return (obs - obs_mean) / obs_std


class RunningObsNormalizer(torch.nn.Module):
    """Online mean/std obs normalizer updated incrementally during rollout.

    Unlike ``ObsNormalizingMixin`` (fit once from an offline dataset, then
    frozen), this tracks a running estimate that keeps moving throughout
    training -- ports rsl_rl's ``EmpiricalNormalization``
    (``3rd_party/rsl_rl/rsl_rl/modules/normalization.py``). ``update()`` only
    updates statistics in ``training`` mode, so calling it during an eval
    rollout (``policy.eval()`` is active) is automatically a no-op.

    KNOWN LIMITATION under multi-GPU DDP (``rl_garden/common/ddp.py``):
    ``update()`` is never all-reduced across ranks, so each rank's stats
    drift apart from the others over training even though the policy
    weights stay in lockstep (via gradient all-reduce). Rank-0-only
    checkpointing therefore saves rank 0's own local view of these stats,
    not a globally-averaged one. Correctly synchronizing this needs a
    parallel-variance (Welford) merge across ranks, not a plain average --
    not implemented here.
    """

    def __init__(self, dim: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)
