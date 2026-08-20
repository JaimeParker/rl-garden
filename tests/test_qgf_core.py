from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.qgf import QGF


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> QGF:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
        denoise_steps=4,
    )
    defaults.update(kwargs)
    return QGF(**defaults)


def _fill(agent: QGF, steps: int = 64) -> None:
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
        QGF(env=unsupported, buffer_device="cpu", device="cpu")


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "value_loss", "bc_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_all_networks_update_every_step():
    agent = _make_agent()
    _fill(agent)

    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]
    value_before = [p.clone() for p in agent.policy.value.parameters()]

    agent.train(1)

    assert not all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(value_before, agent.policy.value.parameters())
    )


def test_q_agg_min_is_default_and_not_silently_ignored():
    agent = _make_agent()
    assert agent.q_agg == "min"
    q_all = torch.tensor([[[1.0], [2.0]], [[3.0], [0.5]]])
    assert torch.equal(agent.policy._aggregate_q(q_all), q_all.min(dim=0).values)

    agent.q_agg = "mean"
    agent.policy.q_agg = "mean"
    assert torch.equal(agent.policy._aggregate_q(q_all), q_all.mean(dim=0))


def test_predict_in_bounds_for_every_sampling_mode():
    for mode in ("guided", "grad_step", "best_of_n"):
        agent = _make_agent(sampling_mode=mode, actor_num_samples=4)
        obs = torch.randn(2, *agent.env.single_observation_space.shape)
        action = agent.policy.predict(obs)
        assert action.shape == (2, 3)
        assert torch.all(action >= agent.policy.action_low)
        assert torch.all(action <= agent.policy.action_high)


def test_zero_guidance_weight_reduces_to_plain_bc_denoise():
    """guidance_weight=0 must make the guided path numerically identical to
    a plain BC-only Euler denoise -- the cheapest decisive correctness check
    for the guided-sampling loop."""
    agent = _make_agent(sampling_mode="guided")
    agent.policy.guidance_weight = 0.0
    obs = torch.randn(4, *agent.env.single_observation_space.shape)

    torch.manual_seed(0)
    guided_action = agent.policy.predict(obs)

    torch.manual_seed(0)
    features = agent.policy.extract_features(obs)
    x0 = torch.randn(4, agent.policy.actor.action_dim)
    with torch.no_grad():
        plain_bc_action = agent.policy._bc_denoise(features, x0)

    assert torch.allclose(guided_action, plain_bc_action, atol=1e-5)


def test_guided_predict_leaves_no_grad_on_any_network_parameter():
    """The critical correctness trap for gradient-at-inference: the guidance
    gradient must come from torch.autograd.grad, never .backward(), so it
    never accumulates into critic/value/actor .grad and silently corrupts a
    later training step."""
    agent = _make_agent(sampling_mode="guided")
    obs = torch.randn(4, *agent.env.single_observation_space.shape)
    agent.policy.predict(obs)

    for p in agent.policy.parameters():
        assert p.grad is None


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


def test_horizon_length_greater_than_one_uses_chunked_buffer():
    agent = _make_agent(horizon_length=3)
    assert agent.policy.action_low.shape == (9,)
    _fill(agent, steps=64)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "value_loss", "bc_loss"):
        assert np.isfinite(metrics[key])


def test_ifql_faithful_t_sampling_is_continuous_not_grid():
    agent = _make_agent(t_sampling="uniform")
    t = agent._sample_flow_time(1000, device=torch.device("cpu"), dtype=torch.float32)
    # A discrete grid over denoise_steps=4 would only ever produce 5 unique
    # values; continuous uniform sampling should produce far more.
    assert t.unique().numel() > 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("horizon_length", [1, 3])
@pytest.mark.parametrize("sampling_mode", ["guided", "grad_step", "best_of_n"])
def test_cuda_smoke_all_sampling_modes_and_horizons(horizon_length, sampling_mode):
    agent = _make_agent(
        env=_state_env(),
        buffer_device="cuda",
        device="cuda",
        horizon_length=horizon_length,
        sampling_mode=sampling_mode,
        actor_num_samples=4,
    )
    _fill(agent, steps=256)
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    # train()'s own last backward pass legitimately leaves .grad populated
    # (optimizer.step() doesn't clear it) -- zero explicitly so predict()'s
    # effect on .grad can be isolated and checked below.
    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3 * horizon_length)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)
    for p in agent.policy.parameters():
        assert p.grad is None
