from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.networks import ConditionalVAE


def _asymmetric_action_space() -> spaces.Box:
    return spaces.Box(
        low=np.array([-2.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 5.0], dtype=np.float32),
        dtype=np.float32,
    )


def _vae(state_dim: int = 6, **kwargs) -> ConditionalVAE:
    return ConditionalVAE(state_dim, _asymmetric_action_space(), hidden_dim=16, **kwargs)


def test_encode_decode_shapes():
    vae = _vae()
    state = torch.randn(8, 6)
    action = torch.rand(8, 2)
    mean, std = vae.encode(state, action)
    assert mean.shape == (8, vae.latent_dim)
    assert std.shape == (8, vae.latent_dim)
    assert torch.all(std > 0)

    z = torch.randn(8, vae.latent_dim)
    recon = vae.decode(state, z)
    assert recon.shape == (8, 2)


def test_forward_reconstruction_shape():
    vae = _vae()
    state = torch.randn(4, 6)
    action = torch.rand(4, 2)
    recon, mean, std = vae.forward(state, action)
    assert recon.shape == action.shape
    assert mean.shape == std.shape == (4, vae.latent_dim)


def test_decode_respects_asymmetric_action_bounds():
    vae = _vae()
    state = torch.randn(64, 6)
    z = torch.randn(64, vae.latent_dim) * 100.0
    recon = vae.decode(state, z)
    assert torch.all(recon[:, 0] >= -2.0 - 1e-4) and torch.all(recon[:, 0] <= 1.0 + 1e-4)
    assert torch.all(recon[:, 1] >= 0.0 - 1e-4) and torch.all(recon[:, 1] <= 5.0 + 1e-4)


def test_decode_samples_from_prior_when_z_is_none():
    """BCQ/PLAS's candidate-action-sampling path -- SPOT never calls this."""
    vae = _vae()
    state = torch.randn(1000, 6)
    recon = vae.decode(state)
    assert recon.shape == (1000, 2)
    assert torch.all(recon[:, 0] >= -2.0 - 1e-4) and torch.all(recon[:, 0] <= 1.0 + 1e-4)
    assert torch.all(recon[:, 1] >= 0.0 - 1e-4) and torch.all(recon[:, 1] <= 5.0 + 1e-4)


def test_decode_none_z_is_stochastic_across_calls():
    vae = _vae()
    state = torch.randn(8, 6)
    torch.manual_seed(0)
    recon_a = vae.decode(state)
    recon_b = vae.decode(state)
    assert not torch.allclose(recon_a, recon_b)


def test_decode_none_respects_clip():
    """BCQ hardcodes clip=0.5; PLAS exposes it as an optional decode() arg."""
    vae = _vae()
    state = torch.randn(2000, 6)
    torch.manual_seed(0)
    unclipped_recon = vae.decode(state)
    torch.manual_seed(0)
    clipped_recon = vae.decode(state, clip=0.5)
    assert not torch.allclose(unclipped_recon, clipped_recon)


def test_decode_none_uses_state_device_not_a_stored_device_field():
    """Regression guard: CORL's own copy of this branch references an
    undefined self.device (a bug introduced when adapting BCQ's code, which
    correctly stores a device field). This class must not require one."""
    vae = _vae()
    assert not hasattr(vae, "device")
    state = torch.randn(4, 6)
    recon = vae.decode(state)
    assert recon.device == state.device


def test_loss_shape_and_finite_grad():
    vae = _vae()
    state = torch.randn(8, 6)
    action = torch.rand(8, 2)
    losses = vae.loss(state, action, beta=0.5)
    assert set(losses) == {"vae_loss", "recon_loss", "kl_loss"}
    for value in losses.values():
        assert value.shape == ()
    losses["vae_loss"].backward()
    grads = [p.grad for p in vae.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_default_latent_dim_is_twice_action_dim():
    vae = _vae()
    assert vae.latent_dim == 4


def test_explicit_latent_dim_overrides_default():
    vae = _vae(latent_dim=7)
    assert vae.latent_dim == 7
