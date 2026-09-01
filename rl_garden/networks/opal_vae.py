"""OPAL's obs-conditioned skill VAE (Ajay et al. 2021, "OPAL: Offline
Primitive Discovery for Accelerating Offline Reinforcement Learning").

Ported directly from ``SUPE/supe/pretraining/opal.py::VAE`` (read
in full), state-obs path only -- the reference's CNN/pixel branch
(``opal.py:154-214``) is out of scope, matching every other state-only phase
in this port. Two details read as inconsistencies but are **deliberate**,
matching the reference exactly -- do not "fix" them:

1. The posterior encoder's second output half is **log-variance**, not
   log-std: ``std = exp(0.5 * log_var)`` (``opal.py:249``). The prior and
   decoder heads (both a JAX ``GaussianModule``) instead use the standard
   log-std convention, ``std = exp(log_std)`` (``opal.py:130``). Using
   ``exp(x)`` for the posterior would silently square the true std -- it
   still trains, so this is the single highest-risk bug in this port.
2. The posterior's log-variance is never clipped (``opal.py:246-249`` has
   no clip anywhere in that block), unlike the prior/decoder's ``log_std``,
   which is clamped to ``[-20, 2]`` inside ``GaussianModule`` itself
   (``opal.py:127``). Ported as two independently-behaving heads, not a
   single shared clip.

The decoder reuses ``UnsquashedGaussianActor`` verbatim: OPAL's decoder
(``recon_model``, ``opal.py:175``) is exactly a diagonal-Gaussian MLP head
over ``(obs, z) -> action``, clipped to the action-space bounds only at
*sampling* time (never inside the module itself) -- precisely
``UnsquashedGaussianActor``'s existing contract, including its
``evaluate_action_log_prob`` for the reconstruction NLL term.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.networks.actor_critic import UnsquashedGaussianActor
from rl_garden.networks.bigru_sequence_encoder import BiGRUSequenceEncoder
from rl_garden.networks.mlp import create_mlp


class OPALVAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: spaces.Box,
        skill_dim: int,
        chunk_size: int,
        *,
        hidden_size: int = 256,
        prior_hidden_dims: Sequence[int] = (256, 256),
        decoder_hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        action_dim = int(action_space.shape[0])
        self.skill_dim = skill_dim
        self.chunk_size = chunk_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Per-timestep obs embedding (opal.py:78,89-92): a plain 2-layer MLP
        # with activation after *both* layers (matches upstream's
        # `MLP([h, h], activate_final=True)` -- `create_mlp`'s `output_dim=-1`
        # already activates every `net_arch` entry, no special-casing needed).
        self.obs_mlp = create_mlp(
            obs_dim, -1, [hidden_size, hidden_size],
            activation_fn=nn.ReLU, kernel_init="xavier_uniform",
        )
        self.posterior_encoder = BiGRUSequenceEncoder(
            input_dim=hidden_size + action_dim,
            seq_len=chunk_size,
            latent_dim=2 * skill_dim,
            hidden_size=hidden_size,
            num_layers=2,
        )

        # Learned, obs-conditioned prior p(z|s_0) (opal.py:173,252-254) --
        # NOT a fixed N(0, I). Conditions on the chunk's raw first
        # observation, not the obs_mlp embedding above.
        prior_hidden_dims = list(prior_hidden_dims)
        self.prior_trunk = create_mlp(
            obs_dim, -1, prior_hidden_dims,
            activation_fn=nn.ReLU, kernel_init="xavier_uniform",
        )
        self.prior_mean = nn.Linear(prior_hidden_dims[-1], skill_dim)
        self.prior_log_std = nn.Linear(prior_hidden_dims[-1], skill_dim)

        self.decoder = UnsquashedGaussianActor(
            obs_dim + skill_dim, action_space, list(decoder_hidden_dims),
            std_parameterization="exp", log_std_min=log_std_min,
            log_std_max=log_std_max, tanh_mean=False, kernel_init="xavier_uniform",
        )

    def _posterior(
        self, obs_window: torch.Tensor, action_window: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_embed = self.obs_mlp(obs_window)  # (B, C, hidden_size)
        seq_in = torch.cat([obs_embed, action_window], dim=-1)
        out = self.posterior_encoder(seq_in)  # (B, 2*skill_dim)
        mean, log_var = out[..., : self.skill_dim], out[..., self.skill_dim :]
        # Log-variance, NOT log-std -- and deliberately unclipped. See the
        # module docstring's asymmetry note.
        std = torch.exp(0.5 * log_var)
        return mean, std

    def _prior(self, obs_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.prior_trunk(obs_0)
        mean = self.prior_mean(h)
        log_std = self.prior_log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mean, torch.exp(log_std)

    def encode(self, obs_window: torch.Tensor, action_window: torch.Tensor) -> torch.Tensor:
        """Mean-only posterior skill embedding (matches upstream's
        ``encode()``, ``opal.py:216-219`` -- used for offline skill-labeling
        and online re-encoding, no sampling)."""
        mean, _ = self._posterior(obs_window, action_window)
        return mean

    def loss(
        self, obs_window: torch.Tensor, action_window: torch.Tensor, *, kl_coef: float
    ) -> dict[str, torch.Tensor]:
        mean, std = self._posterior(obs_window, action_window)
        posterior = torch.distributions.Normal(mean, std)
        prior_mean, prior_std = self._prior(obs_window[:, 0])
        prior = torch.distributions.Normal(prior_mean, prior_std)

        z = posterior.rsample()
        chunk_size = obs_window.shape[1]
        z_expand = z.unsqueeze(1).expand(-1, chunk_size, -1)
        features = torch.cat([obs_window, z_expand], dim=-1)
        recon_log_prob = self.decoder.evaluate_action_log_prob(features, action_window)
        recon_loss = -recon_log_prob.mean()

        kl = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
        total_loss = recon_loss + kl_coef * kl
        return {
            "vae_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl,
            "prior_std": prior_std.mean(),
            "posterior_std": std.mean(),
        }
