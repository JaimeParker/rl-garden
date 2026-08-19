"""DDPM forward/reverse process math shared by ``DiffusionPolicy`` (BC
pretrain) and ``DPPOPolicy`` (PPO fine-tune).

Ported from ``3rd_party/dppo/model/diffusion/diffusion.py::DiffusionModel``
and ``model/diffusion/sampling.py``, verified against source directly.
**DDPM only** -- the reference's DDIM branch (``use_ddim``, learnable
``eta``) is out of scope for this port and intentionally not implemented.

A plain mixin, not an ``nn.Module`` itself: buffers are registered onto
whichever concrete policy (``nn.Module``, via ``BasePolicy``) mixes this in,
so ``.to(device)`` on the policy moves them -- same mixin shape as
``TD3BCCore``/``FQLCore`` for algorithms. Call ``_init_diffusion_process``
after the concrete class's ``super().__init__()``.

The reverse sampler (``sample_chain``) takes epsilon-prediction as an
injected ``predict_noise(x, t) -> noise`` closure rather than owning a
network directly -- this is a deliberate factoring difference from the
reference (whose ``DiffusionModel``/``VPGDiffusion`` own ``self.network``/
``self.actor``+``self.actor_ft`` directly): it lets ``DiffusionPolicy``
(single network) and ``DPPOPolicy`` (frozen ``actor`` + trainable
``actor_ft``, mixed per ``VPGDiffusion.p_mean_var``'s ``ft_indices`` rule)
share this exact loop and posterior math without duplicating either. The
posterior formulas themselves (``q_sample``, ``p_mean_var``'s mu/logvar,
the reverse-sampler step) are unchanged from source.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _cosine_beta_schedule(denoising_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = denoising_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas, dtype=torch.float32)


def _extract(a: torch.Tensor, t: torch.Tensor, x_ndim: int) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *([1] * (x_ndim - 1)))


class DiffusionProcess:
    def _init_diffusion_process(
        self,
        *,
        denoising_steps: int,
        denoised_clip_value: Optional[float] = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: Optional[float] = None,
    ) -> None:
        if denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {denoising_steps}.")
        self.denoising_steps = denoising_steps
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value

        betas = _cosine_beta_schedule(denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        ddpm_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        self.register_buffer(
            "ddpm_logvar_clipped", torch.log(torch.clamp(ddpm_var, min=1e-20))
        )
        self.register_buffer(
            "ddpm_mu_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "ddpm_mu_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """q(x_t | x_0) = N(x_t; sqrt(alphas_cumprod_t) x_0, (1 - alphas_cumprod_t) I)."""
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.dim()) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.dim()) * noise
        )

    def _predict_x0(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        x_recon = (
            _extract(self.sqrt_recip_alphas_cumprod, t, x.dim()) * x
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x.dim()) * noise
        )
        if self.denoised_clip_value is not None:
            x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        return x_recon

    def p_mean_var(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DDPM posterior mean/logvar given an already-computed predicted
        epsilon ``noise`` (frozen/fine-tuned-actor mixing, if any, is the
        caller's responsibility -- see module docstring)."""
        x_recon = self._predict_x0(x, t, noise)
        mu = (
            _extract(self.ddpm_mu_coef1, t, x.dim()) * x_recon
            + _extract(self.ddpm_mu_coef2, t, x.dim()) * x
        )
        logvar = _extract(self.ddpm_logvar_clipped, t, x.dim())
        return mu, logvar

    def p_losses(
        self, network: Callable, x_start: torch.Tensor, cond: dict, t: torch.Tensor
    ) -> torch.Tensor:
        """Epsilon-prediction MSE at random ``t`` -- the BC pretrain loss."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred = network(x_noisy, t, cond=cond)
        return F.mse_loss(pred, noise, reduction="mean")

    def sample_chain(
        self,
        cond: dict,
        *,
        horizon_steps: int,
        action_dim: int,
        predict_noise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        deterministic: bool,
        min_sampling_denoising_std: float,
        return_chain: bool = False,
        chain_start_t: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """DDPM reverse sampler, ``t`` from ``denoising_steps - 1`` down to 0.

        ``chain_start_t``: when set, records the post-step ``x`` for every
        step with ``t <= chain_start_t`` (matches the reference's
        ``t <= ft_denoising_steps`` condition exactly) -- ``chain[0]`` is the
        value produced entering the recorded window, ``chain[-1]`` is the
        final clean action (same tensor as the returned ``x``).
        """
        device = self.betas.device
        batch = next(iter(cond.values())).shape[0]
        x = torch.randn((batch, horizon_steps, action_dim), device=device)
        chain = [] if return_chain else None
        if return_chain and chain_start_t is not None and chain_start_t == self.denoising_steps:
            # Whole chain is fine-tuned: reference pre-appends the initial
            # sample before the loop (diffusion_vpg.py's `forward`) since the
            # in-loop `t_val <= chain_start_t` condition below never fires for
            # t_val == denoising_steps.
            chain.append(x)
        for t_val in range(self.denoising_steps - 1, -1, -1):
            t = torch.full((batch,), t_val, device=device, dtype=torch.long)
            noise = predict_noise(x, t)
            mean, logvar = self.p_mean_var(x, t, noise)
            std = torch.exp(0.5 * logvar)
            if deterministic and t_val == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=min_sampling_denoising_std)
            step_noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * step_noise
            if self.final_action_clip_value is not None and t_val == 0:
                x = x.clamp(-self.final_action_clip_value, self.final_action_clip_value)
            if return_chain and chain_start_t is not None and t_val <= chain_start_t:
                chain.append(x)
        if return_chain:
            chain = torch.stack(chain, dim=1) if chain else None
        return x, chain
