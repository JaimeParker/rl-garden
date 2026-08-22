from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.spot import SPOT


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> SPOT:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
        vae_hidden_dim=16,
    )
    defaults.update(kwargs)
    return SPOT(**defaults)


def _fill(agent: SPOT, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="Box"):
        SPOT(env=unsupported, buffer_device="cpu", device="cpu")


def _pretrained_agent(**kwargs) -> SPOT:
    agent = _make_agent(vae_iterations=5, **kwargs)
    _fill(agent)
    agent.pretrain_vae()
    return agent


def test_gradient_step_produces_finite_losses():
    agent = _pretrained_agent()
    metrics = agent.train(4, compute_info=True)
    for key in ("critic_loss", "actor_loss", "neg_log_beta", "lambd"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_predict_in_bounds():
    agent = _make_agent()
    obs = torch.randn(4, 6)
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_elbo_vs_iwae_selection_changes_actor_loss():
    """Toggling iwae changes which density estimator computes neg_log_beta,
    so the same fixed data/network state must produce different actor
    losses under the two modes."""
    torch.manual_seed(0)
    agent_elbo = _pretrained_agent(iwae=False, policy_freq=1)
    torch.manual_seed(0)
    agent_iwae = _pretrained_agent(iwae=True, policy_freq=1)
    agent_iwae.policy.load_state_dict(agent_elbo.policy.state_dict())

    data = agent_elbo._sample_train_batch(agent_elbo.batch_size)
    features = agent_elbo.policy.extract_features(data.obs)
    pi_action = agent_elbo.policy.actor(features)

    elbo_val = agent_elbo.policy.vae.elbo_loss(features, pi_action, agent_elbo.beta, 1)
    iwae_val = -agent_iwae.policy.vae.iwae_ll(features, pi_action, agent_iwae.beta, 1)
    assert not torch.allclose(elbo_val, iwae_val)


def test_policy_freq_delays_actor_and_target_updates():
    agent = _pretrained_agent(policy_freq=3)
    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    target_before = [p.clone() for p in agent.policy.actor_target.parameters()]

    agent.train(2)  # 2 < policy_freq=3, actor/target should not move

    assert all(torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters()))
    assert all(
        torch.equal(a, b) for a, b in zip(target_before, agent.policy.actor_target.parameters())
    )

    agent.train(1)  # 3rd step triggers the delayed update
    assert not all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )


def test_norm_q_has_no_alpha_numerator():
    """Regression guard: SPOT's actor loss uses norm_q = 1/|q|.mean() with no
    alpha numerator (spot.py:634), unlike TD3BCCore's lmbda = alpha/|q|.mean().
    Two agents with identical policy/buffer state but wildly different
    (inherited, unused) `alpha` must produce bit-identical actor_loss when
    each `train(1)` call is run from the same RNG seed -- if `alpha` were
    ever reintroduced into the norm_q formula, this would diverge."""
    agent_a = _pretrained_agent(policy_freq=1)
    agent_b = _make_agent(policy_freq=1, vae_iterations=0)
    agent_b.replay_buffer = agent_a.replay_buffer
    agent_b.policy.load_state_dict(agent_a.policy.state_dict())
    agent_b._vae_pretrained = True
    for p in agent_b.policy.vae.parameters():
        p.requires_grad_(False)
    agent_b.policy.vae.eval()
    agent_b.alpha = 999.0  # unused inherited TD3BCCore field

    torch.manual_seed(123)
    metrics_a = agent_a.train(1, compute_info=True)
    torch.manual_seed(123)
    metrics_b = agent_b.train(1, compute_info=True)
    assert metrics_a["actor_loss"] == pytest.approx(metrics_b["actor_loss"], rel=1e-6)


def test_vae_frozen_after_pretrain():
    agent = _pretrained_agent()
    for p in agent.policy.vae.parameters():
        assert p.requires_grad is False
    assert agent.policy.vae.training is False

    before = [p.clone() for p in agent.policy.vae.parameters()]
    agent.train(3)
    after = list(agent.policy.vae.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_pretrain_vae_is_idempotent():
    agent = _make_agent(vae_iterations=5)
    _fill(agent)
    agent.pretrain_vae()
    weights_after_first = [p.clone() for p in agent.policy.vae.parameters()]
    result = agent.pretrain_vae()
    assert result == {}
    assert all(
        torch.equal(a, b)
        for a, b in zip(weights_after_first, agent.policy.vae.parameters())
    )


def test_pretrain_vae_does_not_touch_global_update():
    agent = _make_agent(vae_iterations=20)
    _fill(agent)
    assert agent._global_update == 0
    agent.pretrain_vae()
    assert agent._global_update == 0


def test_checkpoint_round_trip():
    import os
    import tempfile

    agent = _pretrained_agent()
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
    assert loaded._vae_pretrained is True
    for p in loaded.policy.vae.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_smoke():
    agent = _make_agent(
        buffer_device="cuda", device="cuda", vae_iterations=5, policy_freq=1
    )
    _fill(agent, steps=256)
    agent.pretrain_vae()
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    for p in agent.policy.parameters():
        assert p.grad is None
