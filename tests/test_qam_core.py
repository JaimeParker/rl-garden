from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.qam import QAM


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> QAM:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=4,
    )
    defaults.update(kwargs)
    return QAM(**defaults)


def _fill(agent: QAM, steps: int = 64) -> None:
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
        QAM(env=unsupported, buffer_device="cpu", device="cpu")


def test_rejects_both_bolt_ons():
    with pytest.raises(ValueError, match="Only one of"):
        _make_agent(fql_alpha=1.0, edit_scale=0.1)


@pytest.mark.parametrize("critic_loss_type", ["ddpg", "iql"])
def test_gradient_step_produces_finite_losses(critic_loss_type):
    agent = _make_agent(critic_loss_type=critic_loss_type)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "flow_loss", "adj_loss", "actor_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    if critic_loss_type == "iql":
        assert "value_loss" in metrics


def test_fql_alpha_bolt_on_finite_losses():
    agent = _make_agent(fql_alpha=1.0)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("fql_distill_loss", "fql_q_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_edit_scale_bolt_on_finite_losses():
    agent = _make_agent(edit_scale=0.1)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("edit_q_loss", "edit_entropy_loss", "edit_alpha_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_edit_alpha_optimizer_updates_edit_alpha_only():
    agent = _make_agent(edit_scale=0.1)
    _fill(agent)
    before = agent.policy.edit_alpha.current_alpha().clone()
    agent.train(3)
    after = agent.policy.edit_alpha.current_alpha()
    assert not torch.equal(before, after)


def test_all_networks_update_every_step():
    agent = _make_agent()
    _fill(agent)
    actor_slow_before = [p.clone() for p in agent.policy.actor_slow.parameters()]
    actor_fast_before = [p.clone() for p in agent.policy.actor_fast.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    agent.train(1)

    assert not all(
        torch.equal(a, b)
        for a, b in zip(actor_slow_before, agent.policy.actor_slow.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(actor_fast_before, agent.policy.actor_fast.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
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


def test_checkpoint_round_trip_with_edit_scale():
    """edit_alpha_optimizer only exists conditionally -- confirm
    _optimizer_names()'s conditional inclusion round-trips correctly."""
    import os
    import tempfile

    agent = _make_agent(edit_scale=0.1)
    _fill(agent)
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent(edit_scale=0.1)
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_horizon_length_greater_than_one_uses_chunked_buffer():
    agent = _make_agent(horizon_length=3)
    assert agent.policy.full_action_dim == 9
    _fill(agent, steps=64)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "flow_loss", "adj_loss"):
        assert np.isfinite(metrics[key])


def test_rho_pessimism_changes_ddpg_target():
    torch.manual_seed(0)
    agent_no_rho = _make_agent(critic_loss_type="ddpg", rho=0.0)
    _fill(agent_no_rho)
    torch.manual_seed(0)
    agent_rho = _make_agent(critic_loss_type="ddpg", rho=1.0)
    _fill(agent_rho)

    metrics_no_rho = agent_no_rho.train(1, compute_info=True)
    metrics_rho = agent_rho.train(1, compute_info=True)
    assert metrics_no_rho["critic_loss"] != pytest.approx(metrics_rho["critic_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("horizon_length", [1, 3])
@pytest.mark.parametrize("critic_loss_type", ["ddpg", "iql"])
@pytest.mark.parametrize("bolt_on", ["none", "fql_alpha", "edit_scale"])
def test_cuda_smoke_all_modes_and_horizons(horizon_length, critic_loss_type, bolt_on):
    kwargs = dict(
        env=_state_env(),
        buffer_device="cuda",
        device="cuda",
        horizon_length=horizon_length,
        critic_loss_type=critic_loss_type,
        net_arch=[16, 16],
        flow_steps=4,
        buffer_size=1000,
        batch_size=32,
    )
    if bolt_on == "fql_alpha":
        kwargs["fql_alpha"] = 1.0
    elif bolt_on == "edit_scale":
        kwargs["edit_scale"] = 0.1
    agent = QAM(**kwargs)
    _fill(agent, steps=256)
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3 * horizon_length)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)
    for p in agent.policy.parameters():
        assert p.grad is None
