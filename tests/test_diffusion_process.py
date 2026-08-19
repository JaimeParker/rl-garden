from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rl_garden.policies._diffusion_process import DiffusionProcess, _cosine_beta_schedule


class _Harness(DiffusionProcess, nn.Module):
    def __init__(self, denoising_steps: int, **kwargs) -> None:
        super().__init__()
        self._init_diffusion_process(denoising_steps=denoising_steps, **kwargs)


def test_cosine_beta_schedule_matches_closed_form():
    denoising_steps = 20
    betas = _cosine_beta_schedule(denoising_steps)
    steps = denoising_steps + 1
    x = np.linspace(0, steps, steps)
    s = 0.008
    acp = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    acp = acp / acp[0]
    expected = np.clip(1 - (acp[1:] / acp[:-1]), a_min=0, a_max=0.999)
    assert torch.allclose(betas, torch.tensor(expected, dtype=torch.float32))
    assert betas.shape == (denoising_steps,)
    assert (betas >= 0).all() and (betas <= 0.999).all()


def test_alphas_cumprod_buffers_consistent():
    harness = _Harness(denoising_steps=50)
    alphas_cumprod = harness.sqrt_alphas_cumprod**2
    assert torch.allclose(
        harness.sqrt_one_minus_alphas_cumprod**2, 1.0 - alphas_cumprod, atol=1e-6
    )
    assert torch.allclose(
        harness.sqrt_recip_alphas_cumprod**2, 1.0 / alphas_cumprod, atol=1e-4
    )
    # monotonically decreasing (more noise added at higher t)
    assert (alphas_cumprod[1:] <= alphas_cumprod[:-1] + 1e-8).all()


def test_predict_x0_recovers_x_start_from_true_noise():
    harness = _Harness(denoising_steps=100, denoised_clip_value=None)
    x_start = torch.randn(4, 3, 2)
    t = torch.randint(0, 100, (4,))
    noise = torch.randn_like(x_start)
    x_noisy = harness.q_sample(x_start, t, noise)
    x_recon = harness._predict_x0(x_noisy, t, noise)
    assert torch.allclose(x_recon, x_start, atol=1e-4)


def test_p_mean_var_matches_hand_derived_coefficients():
    harness = _Harness(denoising_steps=10, denoised_clip_value=None)
    x = torch.randn(3, 2, 2)
    t = torch.tensor([0, 3, 9])
    noise = torch.randn_like(x)
    mu, logvar = harness.p_mean_var(x, t, noise)

    x_recon = (
        harness.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x
        - harness.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
    )
    expected_mu = (
        harness.ddpm_mu_coef1[t].view(-1, 1, 1) * x_recon
        + harness.ddpm_mu_coef2[t].view(-1, 1, 1) * x
    )
    expected_logvar = harness.ddpm_logvar_clipped[t].view(-1, 1, 1)
    assert torch.allclose(mu, expected_mu, atol=1e-5)
    assert torch.allclose(logvar, expected_logvar, atol=1e-5)


def test_sample_chain_shape_and_final_entry_matches_returned_action():
    harness = _Harness(denoising_steps=5)
    net = nn.Linear(4 + 3 * 2, 3 * 2)

    def predict_noise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        t_feat = t.float().view(batch, 1).expand(batch, 4)
        out = net(torch.cat([t_feat, x.reshape(batch, -1)], dim=-1))
        return out.view(batch, 2, 3)

    cond = {"state": torch.randn(6, 1, 4)}
    x, chain = harness.sample_chain(
        cond,
        horizon_steps=2,
        action_dim=3,
        predict_noise=predict_noise,
        deterministic=False,
        min_sampling_denoising_std=0.1,
        return_chain=True,
        chain_start_t=2,
    )
    assert x.shape == (6, 2, 3)
    assert chain.shape == (6, 3, 2, 3)  # ft_denoising_steps(2)+1 entries
    assert torch.equal(chain[:, -1], x)


def test_sample_chain_records_full_length_when_ft_denoising_steps_equals_denoising_steps():
    # ft_denoising_steps == denoising_steps (chain_start_t == denoising_steps)
    # is a legal, validated config (whole chain fine-tuned). The in-loop
    # `t_val <= chain_start_t` condition never fires for t_val ==
    # denoising_steps, so the initial sample must be pre-appended -- matches
    # the reference's `forward` (diffusion_vpg.py).
    harness = _Harness(denoising_steps=4)
    net = nn.Linear(4 + 2, 2)

    def predict_noise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        t_feat = t.float().view(batch, 1).expand(batch, 4)
        out = net(torch.cat([t_feat, x.reshape(batch, -1)], dim=-1))
        return out.view(batch, 1, 2)

    cond = {"state": torch.randn(3, 1, 4)}
    x, chain = harness.sample_chain(
        cond,
        horizon_steps=1,
        action_dim=2,
        predict_noise=predict_noise,
        deterministic=False,
        min_sampling_denoising_std=0.1,
        return_chain=True,
        chain_start_t=4,
    )
    assert x.shape == (3, 1, 2)
    assert chain.shape == (3, 5, 1, 2)  # denoising_steps(4)+1 entries
    assert torch.equal(chain[:, -1], x)


def test_sample_chain_without_chain_recording():
    harness = _Harness(denoising_steps=3)
    net = nn.Linear(4 + 2, 2)

    def predict_noise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        t_feat = t.float().view(batch, 1).expand(batch, 4)
        out = net(torch.cat([t_feat, x.reshape(batch, -1)], dim=-1))
        return out.view(batch, 1, 2)

    cond = {"state": torch.randn(2, 1, 4)}
    x, chain = harness.sample_chain(
        cond,
        horizon_steps=1,
        action_dim=2,
        predict_noise=predict_noise,
        deterministic=True,
        min_sampling_denoising_std=0.0,
        return_chain=False,
    )
    assert x.shape == (2, 1, 2)
    assert chain is None
