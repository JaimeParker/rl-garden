"""SPOT's behavior-density VAE: ``ConditionalVAE`` + an ELBO/IWAE density estimator.

``ConditionalVAE`` (``rl_garden/networks/conditional_vae.py``) hosts the
network shape and pretraining objective shared with BCQ/PLAS (verified
against both projects' official repos). This subclass adds the two
density-estimation methods that are SPOT-specific -- BCQ/PLAS never call
these; they only use ``decode(z=None)`` to sample candidate actions, never
per-batch density estimates of an arbitrary action.

- ``elbo_loss``: a per-batch-element negative-ELBO proxy, used by SPOT's
  actor loss as a "support constraint" (penalizing actions the VAE assigns
  low density to). Ported from CORL's ``SPOT.elbo_loss``.
- ``iwae_ll``: the importance-sampling log-likelihood estimator, an
  alternative to ``elbo_loss`` selected via SPOT's ``iwae`` flag. Ported
  from CORL's ``VAE.importance_sampling_estimator``.
"""
from __future__ import annotations

import math

import torch

from rl_garden.networks.conditional_vae import ConditionalVAE


class BehaviorVAE(ConditionalVAE):
    """``ConditionalVAE`` plus SPOT's ELBO and IWAE log-likelihood estimators."""

    def elbo_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Per-batch-element negative-ELBO proxy: ``recon_loss + beta * KL``.

        Matches ``spot.py``'s ``elbo_loss`` (lines 520-544): the KL term uses
        the un-repeated encoder ``mean``/``std`` (computed once), while the
        reconstruction term averages over ``num_samples`` reparameterized
        draws of ``z``. Both terms reduce to shape ``[B]``.
        """
        mean, std = self.encode(state, action)

        mean_s = mean.unsqueeze(1).expand(-1, num_samples, -1)
        std_s = std.unsqueeze(1).expand(-1, num_samples, -1)
        z = mean_s + std_s * torch.randn_like(std_s)

        state_s = state.unsqueeze(1).expand(-1, num_samples, -1)
        action_s = action.unsqueeze(1).expand(-1, num_samples, -1)
        recon = self.decode(state_s, z)
        recon_loss = ((recon - action_s) ** 2).mean(dim=(1, 2))

        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon_loss + beta * kl_loss

    def iwae_ll(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """IWAE log-marginal-likelihood estimate ``log p_beta(a|s)``, shape ``[B]``.

        Matches ``spot.py``'s ``importance_sampling_estimator`` (lines
        340-373). SPOT's actor loss uses ``-iwae_ll(...)`` (see ``iwae_loss``
        in the reference, which returns ``-ll``).
        """
        mean, std = self.encode(state, action)

        mean_s = mean.unsqueeze(1).expand(-1, num_samples, -1)
        std_s = std.unsqueeze(1).expand(-1, num_samples, -1)
        z = mean_s + std_s * torch.randn_like(std_s)

        state_s = state.unsqueeze(1).expand(-1, num_samples, -1)
        action_s = action.unsqueeze(1).expand(-1, num_samples, -1)
        mean_dec = self.decode(state_s, z)
        std_dec = math.sqrt(beta / 4.0)

        log_qzx = torch.distributions.Normal(mean_s, std_s).log_prob(z)
        log_pz = torch.distributions.Normal(
            torch.zeros_like(z), torch.ones_like(z)
        ).log_prob(z)
        log_pxz = torch.distributions.Normal(mean_dec, std_dec).log_prob(action_s)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        return w.logsumexp(dim=-1) - math.log(num_samples)
