"""State-conditioned action VAE: the shared network shape behind BCQ/PLAS/SPOT.

Verified against full clones of the two original-author reference
implementations (not just CORL's port, and not just fetched raw files):

- `sfujim/BCQ <https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py>`_
  (Fujimoto's official BCQ) -- ``VAE`` class.
- `Wenxuan-Zhou/PLAS <https://github.com/Wenxuan-Zhou/PLAS/blob/main/algos.py>`_
  (Zhou's official PLAS, whose ``algos.py`` module docstring states it is
  based on ``sfujim/BCQ``) -- ``VAE``/``VAEModule``.

Both reference VAEs are architecturally identical to
``rl_garden.networks.behavior_vae.BehaviorVAE`` (already verified against
CORL's SPOT port): a 750-hidden 2-layer encoder producing ``mean``/``log_std``
(clamped ``[-4, 15]``) heads, a 2-layer decoder ending in a squashed,
action-bound-rescaled output, and the same pretraining objective
(``recon_loss + beta * KL_loss``). This class hosts exactly that shared
"vanilla VAE" surface so ``rl_garden.algorithms.bcq``/``rl_garden.algorithms.plas``
and SPOT's ``BehaviorVAE`` can all subclass it instead of duplicating the
network.

One capability that BCQ/PLAS need but SPOT never calls: ``decode(z=None)``,
sampling a candidate action from the latent prior. Both BCQ's actor
(perturbs VAE-sampled candidates, picks the best by Q) and PLAS's actor
(outputs directly into the VAE's latent space) depend on it. CORL's own copy
of this branch references an undefined ``self.device`` (a bug introduced
when adapting BCQ's code, which does correctly store a ``device`` field) --
this class instead draws on ``state.device``, sidestepping that bug entirely
without needing a stored device field.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Optional

from rl_garden.networks.mlp import create_mlp

_LOG_STD_MIN = -4.0
_LOG_STD_MAX = 15.0


class ConditionalVAE(nn.Module):
    """Vanilla state-conditioned action VAE ``q(a|s)``. See module docstring."""

    def __init__(
        self,
        state_dim: int,
        action_space: spaces.Box,
        *,
        hidden_dim: int = 750,
        latent_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        action_dim = int(action_space.shape[0])
        self.action_dim = action_dim
        self.latent_dim = latent_dim if latent_dim is not None else 2 * action_dim

        self.encoder_trunk = create_mlp(
            state_dim + action_dim, -1, [hidden_dim, hidden_dim]
        )
        self.mean_head = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.latent_dim)

        self.decoder = create_mlp(
            state_dim + self.latent_dim,
            action_dim,
            [hidden_dim, hidden_dim],
            squash_output=True,
        )

        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def encode(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(torch.cat([state, action], dim=-1))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, torch.exp(log_std)

    def decode(
        self,
        state: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        clip: Optional[float] = None,
    ) -> torch.Tensor:
        """Decode a latent action. ``z=None`` samples from the prior ``N(0, I)``.

        The prior-sampling path (used by BCQ/PLAS, not by SPOT) matches
        BCQ's ``clip=0.5``-style call sites via the optional ``clip`` bound
        on the sampled ``z`` (PLAS's own signature: ``decode(state, z=None,
        clip=None)``).
        """
        if z is None:
            z = torch.randn(state.shape[0], self.latent_dim, device=state.device)
            if clip is not None:
                z = z.clamp(-clip, clip)
        raw = self.decoder(torch.cat([state, z], dim=-1))
        return raw * self.action_scale + self.action_bias

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mean, std

    def loss(
        self, state: torch.Tensor, action: torch.Tensor, beta: float
    ) -> dict[str, torch.Tensor]:
        """VAE pretraining objective, matching BCQ/PLAS/SPOT identically:
        ``recon_loss + beta * KL_loss`` (BCQ/PLAS hardcode ``beta=0.5``; SPOT
        exposes it as a config field defaulting to the same ``0.5``)."""
        recon, mean, std = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + beta * kl_loss
        return {"vae_loss": vae_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
