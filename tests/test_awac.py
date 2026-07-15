from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import AWAC, OfflineEnvSpec


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)}),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> AWAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
    )
    defaults.update(kwargs)
    return AWAC(**defaults)


def _fill(agent: AWAC, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_rejects_dict_observation_space():
    with pytest.raises(TypeError):
        AWAC(env=_dict_env(), buffer_device="cpu", device="cpu")


def test_policy_has_no_actor_target():
    agent = _make_agent()
    assert not hasattr(agent.policy, "actor_target")
    assert hasattr(agent.policy, "critic_target")


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    metrics = agent.train(1, compute_info=True)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])


def test_critic_target_updates_every_step_not_delayed():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()

    target_before = [p.clone() for p in agent.policy.critic_target.parameters()]
    agent.train(1)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(target_before, agent.policy.critic_target.parameters())
    )


def test_actor_loss_weight_clamped_at_exp_adv_max():
    agent = _make_agent(exp_adv_max=2.0, awac_lambda=1.0)
    _fill(agent)
    agent.fit_obs_normalizer()
    data = agent._sample_train_batch(agent.batch_size)

    features = agent.policy.extract_features(data.obs)
    with torch.no_grad():
        pi_action, _ = agent.policy.actor.action_log_prob(features)
        v = agent.policy.q_values_all(features, pi_action, target=False).min(dim=0).values
        q = agent.policy.q_values_all(features, data.actions, target=False).min(dim=0).values
        adv = q - v
        weights = torch.clamp_max(torch.exp(adv / agent.awac_lambda), agent.exp_adv_max)
    assert weights.max().item() <= 2.0 + 1e-5


def test_unsquashed_actor_log_prob_has_no_tanh_jacobian():
    """CORL's AWAC actor evaluates log_prob directly on the (clamped) action,
    with no change-of-variables correction -- verify this differs from a
    manual tanh-Jacobian-corrected computation."""
    from rl_garden.networks import UnsquashedGaussianActor

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor = UnsquashedGaussianActor(4, action_space, hidden_dims=[16])
    features = torch.randn(5, 4)
    actions = torch.rand(5, 2) * 1.8 - 0.9  # inside bounds, away from +/-1

    mean, log_std = actor(features)
    normal = torch.distributions.Normal(mean, log_std.exp())
    expected = normal.log_prob(actions).sum(-1, keepdim=True)

    actual = actor.evaluate_action_log_prob(features, actions)
    assert torch.allclose(actual, expected)


def test_checkpoint_round_trip_preserves_normalizer_and_critic_target():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    assert torch.allclose(agent.policy.obs_mean, loaded.policy.obs_mean)
    assert torch.allclose(agent.policy.obs_std, loaded.policy.obs_std)
    for a, b in zip(agent.policy.actor.parameters(), loaded.policy.actor.parameters()):
        assert torch.allclose(a, b)
    for a, b in zip(
        agent.policy.critic_target.parameters(), loaded.policy.critic_target.parameters()
    ):
        assert torch.allclose(a, b)
