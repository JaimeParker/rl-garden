from __future__ import annotations

import copy

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import IDQL, OfflineEnvSpec
from rl_garden.policies._diffusion_process import DiffusionProcess


def _state_env(num_envs: int = 2) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**overrides) -> IDQL:
    kwargs = dict(
        env=_state_env(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        critic_hidden_dims=(16, 16),
        value_hidden_dims=(16, 16),
        diffusion_mlp_dims=(16, 16),
        denoising_steps=3,
        n_action_samples=4,
        std_log=False,
    )
    kwargs.update(overrides)
    return IDQL(**kwargs)


def _fill(agent: IDQL, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_value_and_critic_losses_match_iql_formulas():
    agent = _make_agent(expectile=0.7, gamma=0.9)
    obs = torch.randn(4, 4)
    next_obs = torch.randn(4, 4)
    actions = torch.randn(4, 2).clamp(-1, 1)
    rewards = torch.tensor([1.0, -1.0, 0.5, 0.0])
    dones = torch.tensor([0.0, 1.0, 0.0, 0.0])

    class Batch:
        pass

    data = Batch()
    data.obs, data.next_obs = obs, next_obs
    data.actions, data.rewards, data.dones = actions, rewards, dones

    with torch.no_grad():
        features = agent.policy.extract_features(obs)
        target_q = agent.policy.min_q_value(features, actions, target=True)
        values = agent.policy.value(features)
        diff = target_q - values
        expected_value_loss = (
            torch.where(diff > 0, agent.expectile, 1.0 - agent.expectile) * diff.pow(2)
        ).mean()

        next_features = agent.policy.extract_features(next_obs)
        next_v = agent.policy.value(next_features)
        expected_target_q = rewards.unsqueeze(-1) + agent.gamma * (1.0 - dones.unsqueeze(-1)) * next_v

    _, metrics = agent._compute_losses(data)
    assert torch.isclose(torch.tensor(metrics["value_loss"]), expected_value_loss, atol=1e-5)
    assert torch.isclose(
        torch.tensor(metrics["target_q"]), expected_target_q.mean(), atol=1e-5
    )


def test_diffusion_actor_loss_decreases():
    agent = _make_agent()
    obs = torch.randn(16, 4)
    actions = torch.randn(16, 2).clamp(-1, 1)
    weight = torch.ones(16)
    optimizer = torch.optim.Adam(agent.policy.net_parameters(), lr=1e-2)

    first_loss = agent.policy.diffusion_loss(obs, actions, weight=weight).item()
    for _ in range(30):
        loss = agent.policy.diffusion_loss(obs, actions, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    last_loss = agent.policy.diffusion_loss(obs, actions, weight=weight).item()
    assert last_loss < first_loss


def test_actor_objective_dispatch():
    agent = _make_agent()
    adv = torch.tensor([-1.0, 0.0, 1.0])

    agent.actor_objective = "bc"
    assert torch.equal(agent._actor_weight(adv), torch.ones_like(adv))

    agent.actor_objective = "soft_adv"
    expected = torch.where(adv > 0, agent.expectile, 1.0 - agent.expectile)
    assert torch.equal(agent._actor_weight(adv), expected)

    agent.actor_objective = "hard_adv"
    expected = torch.where(adv >= -0.01, 1.0, 0.0)
    assert torch.equal(agent._actor_weight(adv), expected)

    agent.actor_objective = "exp_adv"
    expected = torch.exp(adv * agent.policy_temperature).clamp(max=100.0)
    assert torch.allclose(agent._actor_weight(adv), expected)


def test_predict_deterministic_vs_stochastic():
    agent = _make_agent()
    obs = torch.randn(6, 4)
    det = agent.policy.predict(obs, deterministic=True)
    stoch = agent.policy.predict(obs, deterministic=False)
    assert det.shape == (6, 2)
    assert stoch.shape == (6, 2)
    assert torch.all(det <= 1.0 + 1e-4) and torch.all(det >= -1.0 - 1e-4)
    assert torch.all(stoch <= 1.0 + 1e-4) and torch.all(stoch >= -1.0 - 1e-4)


def test_target_net_polyak_updates_slower_than_critic():
    agent = _make_agent(tau=0.005, actor_tau=0.001)
    critic_init = copy.deepcopy(agent.policy.critic_target.state_dict())
    actor_init = copy.deepcopy(agent.policy.target_net.state_dict())
    _fill(agent)

    agent.train(10)

    critic_dist = sum(
        (v - critic_init[k]).pow(2).sum().item()
        for k, v in agent.policy.critic_target.state_dict().items()
    )
    actor_dist = sum(
        (v - actor_init[k]).pow(2).sum().item()
        for k, v in agent.policy.target_net.state_dict().items()
    )
    assert actor_dist < critic_dist


def test_beta_schedule_dispatch():
    class P(torch.nn.Module, DiffusionProcess):
        def __init__(self, schedule):
            super().__init__()
            self._init_diffusion_process(denoising_steps=5, schedule=schedule)

    cosine_betas = P("cosine").betas
    vp_betas = P("vp").betas
    linear_betas = P("linear").betas

    assert not torch.allclose(cosine_betas, vp_betas)
    assert torch.allclose(linear_betas, torch.linspace(1e-4, 2e-2, 5))


def test_checkpoint_roundtrip(tmp_path):
    agent = _make_agent()
    _fill(agent)
    agent.train(5)
    path = tmp_path / "idql.pt"
    agent.save(path)

    loaded = _make_agent()
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
