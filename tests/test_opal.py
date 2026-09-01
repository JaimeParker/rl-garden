from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import OPAL, OfflineEnvSpec
from rl_garden.networks import OPALVAE


def _write_h5(path, num_traj: int = 4, length: int = 20) -> None:
    with h5py.File(path, "w") as f:
        for i in range(num_traj):
            g = f.create_group(f"traj_{i}")
            g.create_dataset(
                "obs", data=np.random.randn(length + 1, 6).astype(np.float32)
            )
            g.create_dataset(
                "actions", data=np.random.uniform(-1, 1, (length, 2)).astype(np.float32)
            )
            g.create_dataset("rewards", data=np.zeros(length, dtype=np.float32))
            terminated = np.zeros(length, dtype=bool)
            terminated[-1] = True
            g.create_dataset("terminated", data=terminated)
            g.create_dataset("truncated", data=np.zeros(length, dtype=bool))


def _make_agent(path, **overrides) -> OPAL:
    env = OfflineEnvSpec(
        spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    kwargs = dict(
        env=env,
        dataset_path=str(path),
        device="cpu",
        batch_size=16,
        skill_dim=4,
        chunk_size=4,
        hidden_size=8,
        vae_hidden_dims=(8, 8),
    )
    kwargs.update(overrides)
    return OPAL(**kwargs)


def _make_vae(obs_dim=6, action_dim=2, skill_dim=4, chunk_size=4) -> OPALVAE:
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return OPALVAE(
        obs_dim, action_space, skill_dim, chunk_size,
        hidden_size=8, prior_hidden_dims=(8, 8), decoder_hidden_dims=(8, 8),
    )


def test_encoder_std_is_exp_half_log_var_not_exp_log_std():
    torch.manual_seed(0)
    vae = _make_vae()
    obs_window = torch.randn(5, 4, 6)
    action_window = torch.randn(5, 4, 2)

    obs_embed = vae.obs_mlp(obs_window)
    seq_in = torch.cat([obs_embed, action_window], dim=-1)
    out = vae.posterior_encoder(seq_in)
    mean_expected = out[..., :4]
    log_var = out[..., 4:]
    std_via_half_log_var = torch.exp(0.5 * log_var)
    std_via_full_log_var = torch.exp(log_var)

    mean, std = vae._posterior(obs_window, action_window)
    assert torch.equal(mean, mean_expected)
    assert torch.allclose(std, std_via_half_log_var)
    # exp(0.5*x) != exp(x) whenever x != 0 -- confirm the two formulas
    # actually diverge for this random init, so this test would fail if the
    # implementation used exp(log_var) instead of exp(0.5*log_var).
    assert not torch.allclose(log_var, torch.zeros_like(log_var), atol=1e-3)
    assert not torch.allclose(std, std_via_full_log_var)


def test_posterior_log_var_is_not_clipped_but_prior_log_std_is():
    vae = _make_vae()
    obs_window = torch.randn(3, 4, 6)
    action_window = torch.randn(3, 4, 2)

    # Force the posterior projection to output an extreme log-variance.
    with torch.no_grad():
        vae.posterior_encoder.proj.weight.zero_()
        vae.posterior_encoder.proj.bias[4:] = 100.0  # log_var = 100
    _, std = vae._posterior(obs_window, action_window)
    expected_unclipped_std = torch.exp(torch.tensor(50.0))  # exp(0.5*100), no clamp
    assert torch.allclose(std, expected_unclipped_std.expand_as(std), rtol=1e-3)

    # Force the prior head to output an extreme log-std; it must be clamped.
    # Zeroing the head's own weight removes any dependence on the trunk
    # output, leaving only the (extreme) bias.
    with torch.no_grad():
        vae.prior_log_std.weight.zero_()
        vae.prior_log_std.bias[:] = 100.0
    _, prior_std = vae._prior(obs_window[:, 0])
    expected_clipped_std = torch.exp(torch.tensor(2.0))  # clamped to log_std_max=2.0
    assert torch.allclose(prior_std, expected_clipped_std.expand_as(prior_std), rtol=1e-3)


def test_encode_is_mean_only():
    vae = _make_vae()
    obs_window = torch.randn(3, 4, 6)
    action_window = torch.randn(3, 4, 2)
    mean, _ = vae._posterior(obs_window, action_window)
    encoded = vae.encode(obs_window, action_window)
    assert torch.equal(encoded, mean)
    # Deterministic: repeated calls give the same result (no sampling).
    encoded2 = vae.encode(obs_window, action_window)
    assert torch.equal(encoded, encoded2)


def test_loss_is_nll_plus_kl_divergence():
    vae = _make_vae()
    obs_window = torch.randn(5, 4, 6)
    action_window = torch.rand(5, 4, 2) * 2 - 1  # within [-1, 1]

    mean, std = vae._posterior(obs_window, action_window)
    prior_mean, prior_std = vae._prior(obs_window[:, 0])
    posterior = torch.distributions.Normal(mean, std)
    prior = torch.distributions.Normal(prior_mean, prior_std)

    torch.manual_seed(42)
    z = posterior.rsample()
    chunk_size = obs_window.shape[1]
    z_expand = z.unsqueeze(1).expand(-1, chunk_size, -1)
    features = torch.cat([obs_window, z_expand], dim=-1)
    recon_log_prob = vae.decoder.evaluate_action_log_prob(features, action_window)
    expected_recon_loss = -recon_log_prob.mean()
    expected_kl = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
    expected_total = expected_recon_loss + 0.1 * expected_kl

    torch.manual_seed(42)
    losses = vae.loss(obs_window, action_window, kl_coef=0.1)

    assert torch.isclose(losses["vae_loss"], expected_total, atol=1e-5)
    assert torch.isclose(losses["recon_loss"], expected_recon_loss, atol=1e-5)
    assert torch.isclose(losses["kl_loss"], expected_kl, atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path)
    agent.train(5)
    ckpt_path = tmp_path / "opal.pt"
    agent.save(ckpt_path)

    loaded = _make_agent(path)
    loaded.load(ckpt_path, load_replay_buffer=False)
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_train_reduces_loss(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path, num_traj=8, length=40)
    agent = _make_agent(path)

    first = agent.train(5, compute_info=True)["vae_loss"]
    for _ in range(20):
        last = agent.train(10, compute_info=True)["vae_loss"]
    assert last < first
