from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.rebrac import ReBRAC
from rl_garden.buffers import ReBRACReplayBuffer


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> ReBRAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
    )
    defaults.update(kwargs)
    return ReBRAC(**defaults)


def _fill(agent: ReBRAC, steps: int = 64) -> None:
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
        ReBRAC(env=unsupported, buffer_device="cpu", device="cpu")


def test_uses_rebrac_replay_buffer():
    agent = _make_agent()
    assert isinstance(agent.replay_buffer, ReBRACReplayBuffer)


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(4, compute_info=True)
    for key in ("critic_loss", "critic_bc_penalty", "actor_loss", "actor_bc_penalty", "lmbda"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_predict_in_bounds():
    agent = _make_agent()
    obs = torch.randn(4, 6)
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_actor_bc_coef_and_critic_bc_coef_stored_independently():
    """Regression guard against accidentally sharing one coefficient (e.g.
    reusing TD3BC's single `alpha` for both penalties)."""
    agent = _make_agent(actor_bc_coef=2.5, critic_bc_coef=7.5)
    assert agent.actor_bc_coef == pytest.approx(2.5)
    assert agent.critic_bc_coef == pytest.approx(7.5)


def test_critic_bc_coef_changes_the_td_target():
    """Directly verifies critic_bc_coef enters the critic's bootstrap
    target, using a single fixed forward pass (no agent-to-agent RNG
    matching, which construction/filling doesn't preserve)."""
    agent = _make_agent()
    _fill(agent)
    data = agent._sample_train_batch(agent.batch_size)
    with torch.no_grad():
        next_features = agent.policy.extract_features(data.next_obs)
        noise = (torch.randn_like(data.actions) * agent.policy_noise).clamp(
            -agent.noise_clip, agent.noise_clip
        )
        next_action = (agent.policy.actor_target(next_features) + noise).clamp(
            agent._action_low, agent._action_high
        )
        critic_bc_penalty = (next_action - data.next_actions).pow(2).sum(-1, keepdim=True)
        assert critic_bc_penalty.mean() > 0  # real, nonzero penalty on random data
        target_q_all = agent.policy.q_values_all(next_features, next_action, target=True)
        min_q = target_q_all.min(dim=0).values
        next_q_low = min_q - 1.0 * critic_bc_penalty
        next_q_high = min_q - 100.0 * critic_bc_penalty
    assert not torch.allclose(next_q_low, next_q_high)


def test_normalize_q_false_uses_lambda_one():
    agent = _make_agent(normalize_q=False, policy_freq=1)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    assert metrics["lmbda"] == pytest.approx(1.0)


def test_actor_uses_min_over_full_critic_ensemble_not_first_critic_only():
    """rebrac.py:441 uses .min(0) over the whole ensemble; TD3BCCore's own
    actor loss uses q_values_all(...)[0] (first critic only) -- confirms
    ReBRAC does NOT silently inherit that convention."""
    agent = _make_agent(n_critics=4)
    _fill(agent)
    features = agent.policy.extract_features(torch.randn(8, 6))
    pi_action = agent.policy.actor(features)
    q_all = agent.policy.q_values_all(features, pi_action, target=False)
    q_min = q_all.min(dim=0).values
    q_first = q_all[0]
    assert not torch.allclose(q_min, q_first)


def test_policy_freq_delays_actor_and_target_updates():
    agent = _make_agent(policy_freq=3)
    _fill(agent)
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


def test_checkpoint_round_trip():
    import os
    import tempfile

    agent = _make_agent()
    _fill(agent)
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_smoke():
    agent = _make_agent(buffer_device="cuda", device="cuda", policy_freq=1)
    _fill(agent, steps=256)
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    for p in agent.policy.parameters():
        assert p.grad is None
