from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import FINO, Off2OnFINO, OfflineEnvSpec
from rl_garden.algorithms.fino import FINOCore
from rl_garden.algorithms.fql import FQLCore
from rl_garden.encoders.combined import default_image_encoder_factory

# Small + fast: "gap" pooling (unlike the default "flatten") tolerates tiny
# images without PlainConv's flatten-layer size mismatch. Mirrors
# tests/test_floq_core.py's own vision-test image encoder factory.
_TEST_IMAGE_SIZE = 16
_test_image_encoder_factory = default_image_encoder_factory(
    features_dim=16, plain_conv_pooling="gap"
)

_METRIC_KEYS = (
    "critic_loss",
    "actor_loss",
    "bc_flow_loss",
    "distill_loss",
    "q_loss",
)


def _state_env(num_envs: int = 1, action_dim: int = 3) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        num_envs=num_envs,
    )


def _vision_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=(_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=np.uint8
                ),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> FINO:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=16,
        device="cpu",
        net_arch=[16, 16],
        flow_steps=4,
    )
    defaults.update(kwargs)
    return FINO(**defaults)


def _fill(agent: FINO, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: FINO, steps: int = 64) -> None:
    env = agent.env
    obs_space = env.single_observation_space
    img_shape = (_TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3)
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, *img_shape), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, *obs_space["state"].shape),
        }
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _assert_predict_in_bounds(agent: FINO) -> None:
    obs_space = agent.env.single_observation_space
    obs = {
        "rgb": torch.randint(
            0, 256, (1, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=torch.uint8
        ),
        "state": torch.randn(1, *obs_space["state"].shape),
    }
    for deterministic in (True, False):
        with torch.no_grad():
            action = agent.policy.predict(obs, deterministic=deterministic)
        assert action.shape == (1, 3)
        assert torch.all(action >= agent.policy.action_low)
        assert torch.all(action <= agent.policy.action_high)


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="Box or Dict"):
        FINO(env=unsupported, buffer_device="cpu", device="cpu")


def test_vision_shared_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="shared",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    assert not hasattr(agent.policy, "actor_bc_flow_encoder")
    _assert_predict_in_bounds(agent)


def test_vision_separate_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="separate",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    _assert_predict_in_bounds(agent)


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_actor_and_critic_update_every_step():
    agent = _make_agent()
    _fill(agent)

    bc_flow_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    onestep_before = [p.clone() for p in agent.policy.actor_onestep_flow.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    metrics = agent.train(1, compute_info=True)

    assert "actor_loss" in metrics
    assert not all(
        torch.equal(a, b) for a, b in zip(bc_flow_before, agent.policy.actor_bc_flow.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(onestep_before, agent.policy.actor_onestep_flow.parameters())
    )
    assert not all(
        torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )


def test_checkpoint_round_trip():
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
    # Unlike FloQ/Value Flows, FINO uses a plain FQL critic (no critic-flow
    # replacement network), so `critic`/`critic_target` ARE present.
    assert "critic" in dict(loaded.policy.named_children())
    assert "critic_target" in dict(loaded.policy.named_children())

    metadata = agent._checkpoint_metadata()
    assert metadata["noise_scale"] == agent.noise_scale
    assert metadata["beta"] == agent.beta
    assert metadata["num_samples"] is not None
    assert isinstance(metadata["num_samples"], int)
    assert metadata["num_samples"] == agent.num_samples


def test_off2on_fino_mro_puts_fino_core_before_fql_core():
    mro = Off2OnFINO.__mro__
    assert mro.index(FINOCore) < mro.index(FQLCore)


def test_predict_deterministic_is_per_observation_argmax():
    """With a controlled `_sample_candidates` patch (action_dim >= 5 so
    K > 1), the action returned for observation `i` must be `i`'s own
    argmax-`q` candidate -- guards `sample_actions`'s
    `q.argmax(dim=-1)` + `candidates[arange(B), idx]` gather."""
    agent = _make_agent(env=_state_env(action_dim=5))
    policy = agent.policy
    batch_size, num_samples, action_dim = 4, 5, 5

    candidates = torch.arange(
        batch_size * num_samples * action_dim, dtype=torch.float32
    ).reshape(batch_size, num_samples, action_dim)
    # Distinct, known argmax index per row.
    top_idx = torch.tensor([2, 0, 4, 1])
    q = torch.zeros(batch_size, num_samples)
    for row, idx in enumerate(top_idx):
        q[row, idx] = 10.0 + row

    def fake_sample_candidates(actor_features, q_features):
        return candidates, q

    policy._sample_candidates = fake_sample_candidates

    obs = torch.randn(batch_size, *agent.env.single_observation_space.shape)
    action = policy.sample_actions(obs, deterministic=True)

    expected = candidates[torch.arange(batch_size), top_idx]
    assert torch.equal(action, expected)


def test_sample_candidates_rows_own_their_candidates():
    """Unlike the patched argmax test above, this exercises the real
    `_sample_candidates` implementation end-to-end: guards the
    `repeat_interleave`/`view` alignment (a `.repeat()`-style expansion
    would produce identical shapes/bounds while silently mixing up which
    candidates belong to which observation -- shape/bounds-only assertions
    can't catch that)."""
    agent = _make_agent(num_samples=4)
    policy = agent.policy
    batch_size, num_samples, action_dim = 3, 4, 3
    obs = torch.randn(batch_size, *agent.env.single_observation_space.shape)
    features = policy.extract_features(obs)

    torch.manual_seed(0)
    candidates, q = policy._sample_candidates(features, features)

    torch.manual_seed(0)
    noise = torch.randn(batch_size * num_samples, action_dim)
    expected = (
        policy.actor_onestep_flow(features.repeat_interleave(num_samples, dim=0), noise)
        .clamp(policy.action_low, policy.action_high)
        .view(batch_size, num_samples, action_dim)
    )
    assert torch.allclose(candidates, expected)
    assert q.shape == (batch_size, num_samples)


def test_predict_and_train_with_min_agg():
    """q_agg='min' exercises `_aggregate_q`'s min branch, otherwise dead in
    this suite (every other test uses the "mean" default)."""
    agent = _make_agent(q_agg="min", num_samples=4)
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in _METRIC_KEYS:
        assert np.isfinite(metrics[key]), (key, metrics[key])
    obs = torch.randn(4, *agent.env.single_observation_space.shape)
    with torch.no_grad():
        action = agent.policy.predict(obs, deterministic=True)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)


def test_predict_stochastic_shapes_and_bounds():
    agent = _make_agent(num_samples=5)
    obs = torch.randn(4, *agent.env.single_observation_space.shape)
    with torch.no_grad():
        action = agent.policy.predict(obs, deterministic=False)
    assert action.shape == (4, 3)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)


def test_num_samples_one_degenerates_to_plain_onestep_draw():
    """With `num_samples=1` there is only one candidate per observation, so
    the argmax/Boltzmann-sample is a no-op and `sample_actions`/`predict`
    must return exactly `actor_onestep_flow(features, noise).clamp(...)`
    for the same seeded noise."""
    agent = _make_agent(num_samples=1)
    policy = agent.policy
    batch_size = 3
    obs = torch.randn(batch_size, *agent.env.single_observation_space.shape)

    for deterministic in (True, False):
        torch.manual_seed(0)
        action = policy.predict(obs, deterministic=deterministic)

        torch.manual_seed(0)
        features = policy.extract_features(obs)
        noise = torch.randn(batch_size, 3)
        expected = policy.actor_onestep_flow(features, noise).clamp(
            policy.action_low, policy.action_high
        )

        assert torch.allclose(action, expected)


def test_num_samples_none_resolves_to_reference_formula():
    for action_dim in (3, 7):
        agent = _make_agent(env=_state_env(action_dim=action_dim), num_samples=None)
        assert agent.policy.num_samples == min(10, (action_dim + 1) // 2)
        assert agent.num_samples == agent.policy.num_samples


def test_noise_injection_schedule():
    """Pins the noise-injection schedule's no-op at noise_scale=0 and its
    sign/direction (full-strength at t=1, ~0 at t=0) at noise_scale=0.1,
    isolating a single `_actor_update` call (not a full `train()` step) so
    the only `torch.rand`/`torch.randn`/`torch.randn_like` calls in flight
    are the ones inside it."""
    agent = _make_agent()
    _fill(agent)

    captured = {}
    orig_forward = agent.policy.actor_bc_flow.forward

    def spy_forward(features, x, t):
        # `actor_bc_flow` is also called repeatedly by the distill loss's
        # multi-step teacher unroll (`compute_flow_actions`/
        # `flow_onestep_distill_loss`) -- only the FIRST call per
        # `_actor_update` invocation is the noise-injected bc_flow_loss
        # point; later calls use the teacher's own multi-step `x_t` and a
        # per-integration-step scalar `t` broadcast across the batch.
        if "injected" not in captured:
            captured["injected"] = x.detach().clone()
            captured["t"] = t.detach().clone()
        return orig_forward(features, x, t)

    agent.policy.actor_bc_flow.forward = spy_forward

    orig_rand = torch.rand
    orig_randn = torch.randn
    randn_calls = []

    def spy_randn(*args, **kwargs):
        out = orig_randn(*args, **kwargs)
        randn_calls.append(out.clone())
        return out

    def run_actor_update():
        data = agent.replay_buffer.sample(agent.batch_size)
        obs_features = agent.policy.extract_features(data.obs)
        agent._actor_update(data, obs_features)
        return data

    try:
        # --- noise_scale=0.0: injected input must equal x_t exactly. ---
        agent.noise_scale = 0.0
        torch.rand = orig_rand
        torch.randn = spy_randn
        randn_calls.clear()
        captured.clear()
        data = run_actor_update()

        x_0 = randn_calls[0]
        t = captured["t"]
        x_t = (1 - t) * x_0 + t * data.actions
        assert torch.equal(captured["injected"], x_t)

        # --- noise_scale=0.1 with controlled monotone t and unit noise:
        # |injected - x_t| must increase monotonically with t (pins
        # exp(10*(t-1))'s direction). ---
        agent.noise_scale = 0.1
        batch_size = agent.batch_size

        def spy_rand(*args, **kwargs):
            if args and args[0] == batch_size:
                return torch.linspace(0.0, 1.0, batch_size).unsqueeze(-1)
            return orig_rand(*args, **kwargs)

        orig_randn_like = torch.randn_like

        def spy_randn_like(x, *args, **kwargs):
            return torch.ones_like(x)

        torch.rand = spy_rand
        torch.randn = spy_randn
        torch.randn_like = spy_randn_like
        randn_calls.clear()
        captured.clear()
        try:
            data = run_actor_update()
        finally:
            torch.randn_like = orig_randn_like

        x_0 = randn_calls[0]
        t = captured["t"]
        x_t = (1 - t) * x_0 + t * data.actions
        diff = (captured["injected"] - x_t).abs()
        # t is sorted ascending (linspace), so per-row magnitude must also
        # be non-decreasing (all action dims share the same per-row scale
        # since the injected noise is all-ones).
        per_row = diff.mean(dim=-1)
        assert torch.all(per_row[1:] - per_row[:-1] >= -1e-6)
        assert per_row[-1] > per_row[0]
    finally:
        torch.rand = orig_rand
        torch.randn = orig_randn
        agent.policy.actor_bc_flow.forward = orig_forward
