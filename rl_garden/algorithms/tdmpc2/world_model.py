"""Shared-latent world model, ported from
``3rd_party/tdmpc2/tdmpc2/common/world_model.py``.

Single-task only (no task embedding / action masking -- see the module
docstring of ``agent.py``). The encoder half of upstream's own
``common.layers.enc(cfg)`` is replaced by an rl-garden
``BaseFeaturesExtractor`` (``FlattenExtractor`` for state obs,
``CombinedExtractor`` for pixel+state obs) so this port reuses the project's
existing encoder infrastructure instead of TD-MPC2's own conv/MLP encoder;
the SimNorm-projection half is kept, mapping the extractor's flat features
onto the fixed ``latent_dim`` TD-MPC2's dynamics/reward/Q/pi heads expect.

The target-Q ensemble is a plain ``nn.Module`` submodule (Polyak-updated via
``rl_garden.common.utils.polyak_update``), not upstream's
``TensorDictParams``-based scheme -- this makes it round-trip through
``state_dict()``/``load_state_dict()`` automatically, so no dedicated
checkpoint hook is needed for it (see ``agent.py``'s checkpoint section).
Upstream's separate "detached-parameter" Q view (used so the actor update's
backward pass doesn't touch the critic's parameters) collapses to simply
detaching ``self._Q(...)``'s *output* tensor, since ``QEnsemble`` here has no
functorch/vmap parameter-sharing trick to preserve -- same live parameters,
same values, just no gradient into them.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl_garden.algorithms.tdmpc2 import math_utils
from rl_garden.algorithms.tdmpc2.layers import QEnsemble, RunningScale, SimNorm, mlp
from rl_garden.common.types import Obs
from rl_garden.common.utils import polyak_update
from rl_garden.encoders.base import BaseFeaturesExtractor


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: BaseFeaturesExtractor,
        action_dim: int,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        simnorm_dim: int = 8,
        num_q: int = 5,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        dropout: float = 0.01,
        episodic: bool = False,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tau: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_size = (vmax - vmin) / (num_bins - 1) if num_bins > 1 else 0.0
        self.episodic = episodic

        self.encoder = encoder
        self._latent_proj = nn.Sequential(
            nn.Linear(encoder.features_dim, latent_dim), SimNorm(simnorm_dim)
        )
        self._dynamics = mlp(
            latent_dim + action_dim, 2 * [mlp_dim], latent_dim, act=SimNorm(simnorm_dim)
        )
        self._reward = mlp(latent_dim + action_dim, 2 * [mlp_dim], max(num_bins, 1))
        self._termination = (
            mlp(latent_dim, 2 * [mlp_dim], 1) if episodic else None
        )
        self._pi = mlp(latent_dim, 2 * [mlp_dim], 2 * action_dim)
        self._Q = QEnsemble(latent_dim + action_dim, 2 * [mlp_dim], num_bins, num_q, dropout=dropout)
        self._target_Q = QEnsemble(latent_dim + action_dim, 2 * [mlp_dim], num_bins, num_q, dropout=dropout)
        self._target_Q.load_state_dict(self._Q.state_dict())
        for p in self._target_Q.parameters():
            p.requires_grad_(False)
        self.num_q = num_q
        self.tau = tau

        self.scale = RunningScale(tau=tau)

        self.register_buffer("log_std_min", torch.tensor(log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(log_std_max - log_std_min))

        self._apply_init()

    def _apply_init(self) -> None:
        from rl_garden.algorithms.tdmpc2 import init as tdmpc2_init

        self.apply(tdmpc2_init.weight_init)
        tdmpc2_init.zero_([self._reward[-1].weight])
        for q in self._Q.qs:
            tdmpc2_init.zero_([q[-1].weight])

    def soft_update_target_Q(self) -> None:
        polyak_update(self._Q.parameters(), self._target_Q.parameters(), self.tau)

    def encode(self, obs: Obs) -> torch.Tensor:
        features = self.encoder.extract(obs)
        return self._latent_proj(features)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._reward(torch.cat([z, a], dim=-1))

    def termination(self, z: torch.Tensor, unnormalized: bool = False) -> torch.Tensor:
        assert self._termination is not None, "termination head disabled (episodic=False)"
        out = self._termination(z)
        return out if unnormalized else torch.sigmoid(out)

    def pi(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Samples a squashed action from the Gaussian policy prior."""
        mean, log_std_raw = self._pi(z).chunk(2, dim=-1)
        log_std_ = math_utils.log_std(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = math_utils.gaussian_logprob(eps, log_std_)
        action_dims = eps.shape[-1]

        action = mean + eps * log_std_.exp()
        mean, action, log_prob = math_utils.squash(mean, action, log_prob)

        entropy_scale = action_dims * log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std_,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        """``return_type`` in {"min", "avg", "all"}; ``target`` selects the
        Polyak-averaged ensemble; ``detach`` stops gradient into ``self._Q``'s
        parameters (used for the actor update) without touching values."""
        assert return_type in ("min", "avg", "all")
        za = torch.cat([z, a], dim=-1)
        qnet = self._target_Q if target else self._Q
        out = qnet(za)
        if detach and not target:
            out = out.detach()

        if return_type == "all":
            return out

        qidx = torch.randperm(self.num_q, device=out.device)[:2]
        q_scalar = math_utils.two_hot_inv(out[qidx], self.num_bins, self.vmin, self.vmax)
        if return_type == "min":
            return q_scalar.min(0).values
        return q_scalar.sum(0) / 2
