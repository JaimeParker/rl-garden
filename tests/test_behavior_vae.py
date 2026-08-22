from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.networks import BehaviorVAE


def _asymmetric_action_space() -> spaces.Box:
    return spaces.Box(
        low=np.array([-2.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 5.0], dtype=np.float32),
        dtype=np.float32,
    )


def _vae(state_dim: int = 6, **kwargs) -> BehaviorVAE:
    return BehaviorVAE(state_dim, _asymmetric_action_space(), hidden_dim=16, **kwargs)


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
    """Regression guard for the deliberate CORL divergence: action rescaling
    uses the real action_space bounds (via action_scale/action_bias), not a
    single symmetric max_action float."""
    vae = _vae()
    state = torch.randn(64, 6)
    z = torch.randn(64, vae.latent_dim) * 100.0  # push decoder trunk to saturate tanh
    recon = vae.decode(state, z)
    assert torch.all(recon[:, 0] >= -2.0 - 1e-4) and torch.all(recon[:, 0] <= 1.0 + 1e-4)
    assert torch.all(recon[:, 1] >= 0.0 - 1e-4) and torch.all(recon[:, 1] <= 5.0 + 1e-4)
    # Saturated tanh (+-1) should reach both bound extremes, not just [-1, 1].
    assert recon[:, 1].max() > 1.0 + 1e-3


def test_elbo_loss_shape_and_finite_grad():
    vae = _vae()
    state = torch.randn(8, 6)
    action = torch.rand(8, 2)
    loss = vae.elbo_loss(state, action, beta=0.5, num_samples=4)
    assert loss.shape == (8,)
    loss.mean().backward()
    grads = [p.grad for p in vae.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_iwae_ll_shape_and_finite_grad():
    vae = _vae()
    state = torch.randn(8, 6)
    action = torch.rand(8, 2)
    ll = vae.iwae_ll(state, action, beta=0.5, num_samples=10)
    assert ll.shape == (8,)
    assert torch.isfinite(ll).all()
    (-ll.mean()).backward()
    grads = [p.grad for p in vae.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_default_latent_dim_is_twice_action_dim():
    vae = _vae()
    assert vae.latent_dim == 4


def test_explicit_latent_dim_overrides_default():
    vae = _vae(latent_dim=7)
    assert vae.latent_dim == 7
