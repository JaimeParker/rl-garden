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
@pytest.mark.parametrize(
    "sampling_mode", ["guided", "grad_step", "best_of_n", "bptt", "robust_q"]
)
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


# ---------------------------------------------------------------------------
# BPTT (reconstructed sampling_mode) -- see qgf_policy.py's docstring.
# ---------------------------------------------------------------------------


def test_bptt_predict_in_bounds_and_finite():
    agent = _make_agent(sampling_mode="bptt")
    obs = torch.randn(4, 6)
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.isfinite(action).all()
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)


def test_bptt_grad_matches_manual_autograd():
    """Cross-checks _bptt_grad's torch.autograd.grad-based wiring against an
    independently-constructed grad call for the exact same composed
    function -- the decisive correctness check for the reconstructed
    backprop-through-time mechanism, mirroring the QAM adjoint-matching
    cross-check pattern."""
    agent = _make_agent(sampling_mode="bptt", denoise_steps=4)
    features = agent.policy.extract_features(torch.randn(4, 6))
    a_t = torch.randn(4, 3)
    start_step = 1

    wired_grad = agent.policy._bptt_grad(features, a_t, start_step)

    a_leaf = a_t.clone().requires_grad_(True)
    a_clean = agent.policy._bc_denoise_from(features, a_leaf, start_step)
    q = agent.policy._aggregate_q(
        agent.policy.q_values_all(features, a_clean, target=True)
    ).sum()
    (manual_grad,) = torch.autograd.grad(q, a_leaf)

    assert torch.allclose(wired_grad, manual_grad, atol=1e-5)


def test_bptt_gradient_genuinely_differs_from_one_euler_step_approx():
    """BPTT's whole point is differentiating through v_bc's own dependence
    on the input, which QGF's own "one_euler_step_approx" mode explicitly
    stop-gradients away (qgf_policy.py's _guided_denoise uses
    v_bc.detach()). At the same point, the two gradients should therefore
    generally differ -- confirms BPTT is doing genuinely more than the
    existing approximation, not silently collapsing to it."""
    agent = _make_agent(sampling_mode="bptt", denoise_steps=4)
    features = agent.policy.extract_features(torch.randn(8, 6))
    a = torch.randn(8, 3)
    start_step = 0

    bptt_grad = agent.policy._bptt_grad(features, a, start_step)

    t = torch.full((8, 1), start_step / agent.policy.denoise_steps)
    v_bc = agent.policy.actor(features, a, t)
    a_approx = (a + (1 - t) * v_bc.detach()).clamp(
        agent.policy.action_low, agent.policy.action_high
    )
    approx_grad = agent.policy._q_grad(features, a_approx)

    assert not torch.allclose(bptt_grad, approx_grad, atol=1e-4)


def test_bptt_leaves_no_grad_on_any_parameter():
    agent = _make_agent(sampling_mode="bptt")
    agent.policy.predict(torch.randn(4, 6))
    for p in agent.policy.parameters():
        assert p.grad is None


def test_bptt_gradient_step_produces_finite_losses():
    agent = _make_agent(sampling_mode="bptt")
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "value_loss", "bc_loss"):
        assert np.isfinite(metrics[key])
    # BPTT is inference-only -- confirms zero training-loop changes.
    assert "robust_critic_loss" not in metrics


# ---------------------------------------------------------------------------
# RobustQ (reconstructed sampling_mode) -- see qgf_policy.py's docstring.
# ---------------------------------------------------------------------------


def test_robust_q_builds_robust_critic_only_when_selected():
    agent = _make_agent(sampling_mode="robust_q")
    assert agent.policy.robust_critic is not None
    assert agent.policy.robust_time_embed is not None
    guided_agent = _make_agent(sampling_mode="guided")
    assert guided_agent.policy.robust_critic is None


def test_robust_q_predict_in_bounds_and_finite():
    agent = _make_agent(sampling_mode="robust_q")
    obs = torch.randn(4, 6)
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.isfinite(action).all()
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)


def test_robust_q_training_produces_finite_losses_including_robust_critic():
    agent = _make_agent(sampling_mode="robust_q")
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "value_loss", "bc_loss", "robust_critic_loss", "robust_q_mean"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_robust_q_robust_critic_updates_every_step():
    agent = _make_agent(sampling_mode="robust_q")
    _fill(agent)
    before = [p.clone() for p in agent.policy.robust_critic.parameters()]
    agent.train(1)
    after = list(agent.policy.robust_critic.parameters())
    assert not all(torch.equal(a, b) for a, b in zip(before, after))


def test_robust_q_checkpoint_round_trip_with_conditional_optimizer():
    """robust_critic_optimizer only exists when sampling_mode="robust_q" --
    confirms _optimizer_names()'s conditional third entry round-trips."""
    import os
    import tempfile

    agent = _make_agent(sampling_mode="robust_q")
    _fill(agent)
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent(sampling_mode="robust_q")
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_robust_q_leaves_no_grad_on_any_parameter():
    agent = _make_agent(sampling_mode="robust_q")
    _fill(agent)
    agent.train(1)
    agent.policy.zero_grad(set_to_none=True)
    agent.policy.predict(torch.randn(4, 6))
    for p in agent.policy.parameters():
        assert p.grad is None


def test_robust_q_critic_loss_does_not_route_through_grid_gated_t_sampling(monkeypatch):
    """robust_q.py's own robust_critic_loss always uses continuous-uniform
    t, independent of QGF's t_sampling knob (which only gates the main
    policy_loss's noising, not robust_critic_loss's) -- regression guard:
    _robust_critic_loss must never call _sample_flow_time."""
    agent = _make_agent(sampling_mode="robust_q", t_sampling="grid", denoise_steps=4)
    _fill(agent, steps=256)
    data = agent._sample_train_batch(32)

    def _boom(*args, **kwargs):
        raise AssertionError("_robust_critic_loss must not call _sample_flow_time")

    monkeypatch.setattr(agent, "_sample_flow_time", _boom)
    loss, info = agent._robust_critic_loss(data)
    assert np.isfinite(float(loss.item()))
