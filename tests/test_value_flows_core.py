from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import Off2OnValueFlows, OfflineEnvSpec, ValueFlows
from rl_garden.algorithms.fql import FQLCore
from rl_garden.algorithms.value_flows import ValueFlowsCore
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.networks.value_flow_field import (
    ValueFlowVectorField,
    integrate_returns,
    integrate_returns_with_jvp,
)

# Small + fast: "gap" pooling (unlike the default "flatten") tolerates tiny
# images without PlainConv's flatten-layer size mismatch. Mirrors
# tests/test_floq_core.py's own vision-test image encoder factory.
_TEST_IMAGE_SIZE = 16
_test_image_encoder_factory = default_image_encoder_factory(
    features_dim=16, plain_conv_pooling="gap"
)

_METRIC_KEYS = (
    "critic_loss",
    "bcfm_loss",
    "dcfm_loss",
    "weight",
    "q",
    "actor_loss",
    "bc_flow_loss",
    "distill_loss",
    "q_loss",
)


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
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


def _make_agent(**kwargs) -> ValueFlows:
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
    return ValueFlows(**defaults)


def _fill(agent: ValueFlows, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: ValueFlows, steps: int = 64) -> None:
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


def _assert_predict_in_bounds(agent: ValueFlows) -> None:
    obs_space = agent.env.single_observation_space
    obs = {
        "rgb": torch.randint(
            0, 256, (1, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=torch.uint8
        ),
        "state": torch.randn(1, *obs_space["state"].shape),
    }
    with torch.no_grad():
        action = agent.policy.predict(obs)
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
        ValueFlows(env=unsupported, buffer_device="cpu", device="cpu")


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


def test_gradient_step_produces_finite_losses_with_min_agg():
    """q_agg/ret_agg='min' exercises `aggregate(..., "min")`, otherwise dead
    in this suite (every other test uses the "mean" defaults)."""
    agent = _make_agent(q_agg="min", ret_agg="min")
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_target_flows_frozen_and_disjoint_from_critic_optimizer():
    agent = _make_agent()
    _fill(agent)

    for p in agent.policy.critic_flows_target.parameters():
        assert not p.requires_grad

    target_ptrs = {p.data_ptr() for p in agent.policy.critic_flows_target.parameters()}
    critic_optimizer_ptrs = {
        p.data_ptr() for group in agent.critic_optimizer.param_groups for p in group["params"]
    }
    assert target_ptrs.isdisjoint(critic_optimizer_ptrs)

    target_before = [p.clone() for p in agent.policy.critic_flows_target.parameters()]
    agent.train(1)
    # critic_flows_target moves only via polyak update in _update_targets,
    # never via a gradient step, but should still differ after one full
    # train() call.
    assert not all(
        torch.equal(a, b)
        for a, b in zip(target_before, agent.policy.critic_flows_target.parameters())
    )


def test_actor_and_critic_flows_update_every_step():
    agent = _make_agent()
    _fill(agent)

    bc_flow_before = [p.clone() for p in agent.policy.actor_bc_flow.parameters()]
    onestep_before = [p.clone() for p in agent.policy.actor_onestep_flow.parameters()]
    critic_flows_before = [p.clone() for p in agent.policy.critic_flows.parameters()]

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
        torch.equal(a, b)
        for a, b in zip(critic_flows_before, agent.policy.critic_flows.parameters())
    )


@pytest.mark.parametrize("use_layer_norm", [False, True])
def test_jvp_matches_finite_difference(use_layer_norm):
    """tangent ≈ (integrate(x0+eps) - integrate(x0-eps)) / 2eps on a tiny net.
    Parametrized over layer norm since critic_use_layer_norm=True is the
    default in every real config -- LayerNorm's forward-mode JVP rule is
    non-trivial enough to be worth covering explicitly here, not just via
    the finite-loss smoke tests."""
    torch.manual_seed(0)
    net = ValueFlowVectorField(4, 2, [8], use_layer_norm=use_layer_norm)
    net.eval()
    batch_size = 5
    features = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    x0 = torch.randn(batch_size, 1)

    with torch.no_grad():
        _, tangent = integrate_returns_with_jvp(net, features, actions, x0, steps=8)

        eps = 1e-3
        plus = integrate_returns(net, features, actions, x0 + eps, steps=8)
        minus = integrate_returns(net, features, actions, x0 - eps, steps=8)
        finite_diff = (plus - minus) / (2 * eps)

    assert torch.allclose(tangent, finite_diff, atol=1e-2, rtol=1e-2)


def test_integrate_returns_end_times_one_equals_default():
    torch.manual_seed(0)
    net = ValueFlowVectorField(4, 2, [8], use_layer_norm=False)
    net.eval()
    batch_size = 5
    features = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    x0 = torch.randn(batch_size, 1)

    with torch.no_grad():
        default_result = integrate_returns(net, features, actions, x0, steps=6)
        explicit_result = integrate_returns(
            net, features, actions, x0, steps=6, end_times=torch.ones(batch_size, 1)
        )

    assert torch.allclose(default_result, explicit_result)


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
    assert "critic_flows" in dict(loaded.policy.named_children())
    assert "critic_flows_target" in dict(loaded.policy.named_children())
    assert not hasattr(loaded.policy, "critic")
    assert not hasattr(loaded.policy, "critic_target")

    metadata = agent._checkpoint_metadata()
    for key in (
        "min_reward",
        "max_reward",
        "ret_agg",
        "confidence_weight_temp",
        "dcfm_lambda",
        "bcfm_lambda",
        "clip_flow_returns",
    ):
        assert key in metadata


def test_actor_update_clamps_student_action_and_passes_bounds_to_distill_loss():
    """`_actor_update` must clamp `actor_actions` to the policy's action
    bounds before the distill loss (and pass those bounds through so the
    teacher target is clamped too), not only on the later Q term."""
    import rl_garden.algorithms.value_flows as value_flows_module

    agent = _make_agent()
    _fill(agent)

    captured = {}
    original = value_flows_module.flow_onestep_distill_loss

    def spy(teacher, student_action, features, noise, num_steps, *, low=None, high=None):
        captured["student_action"] = student_action.detach().clone()
        captured["low"] = low
        captured["high"] = high
        return original(
            teacher, student_action, features, noise, num_steps, low=low, high=high
        )

    value_flows_module.flow_onestep_distill_loss = spy
    try:
        agent.train(1)
    finally:
        value_flows_module.flow_onestep_distill_loss = original

    assert "student_action" in captured
    assert torch.equal(captured["low"], agent.policy.action_low)
    assert torch.equal(captured["high"], agent.policy.action_high)
    assert torch.all(captured["student_action"] >= agent.policy.action_low)
    assert torch.all(captured["student_action"] <= agent.policy.action_high)


def test_off2on_value_flows_mro_puts_value_flows_core_before_fql_core():
    mro = Off2OnValueFlows.__mro__
    assert mro.index(ValueFlowsCore) < mro.index(FQLCore)


def test_sample_actions_rs_shapes_and_bounds():
    agent = _make_agent(num_samples=5)
    policy = agent.policy
    batch_size = 4
    features = torch.randn(batch_size, agent.policy.features_extractor.features_dim)

    actions = policy.sample_actions_rs(
        features,
        features,
        num_samples=5,
        clip_range=agent.return_clip_range,
        agg=agent.q_agg,
    )
    assert actions.shape == (batch_size, 3)
    assert torch.all(actions >= policy.action_low)
    assert torch.all(actions <= policy.action_high)


def test_sample_actions_rs_num_samples_one_matches_bc_flow_rollout():
    """With `num_samples=1` there is only one candidate per observation, so
    the argmax is a no-op and `sample_actions_rs` must return exactly the
    BC-flow rollout (`compute_flow_actions`) for that same noise -- no
    dependence on the Q-ranking machinery."""
    agent = _make_agent(num_samples=1)
    policy = agent.policy
    batch_size = 3
    features = torch.randn(batch_size, agent.policy.features_extractor.features_dim)

    torch.manual_seed(0)
    actions = policy.sample_actions_rs(
        features,
        features,
        num_samples=1,
        clip_range=agent.return_clip_range,
        agg=agent.q_agg,
    )

    torch.manual_seed(0)
    noise = torch.randn(batch_size, 3)
    expected = policy.compute_flow_actions(features, noise, policy.num_flow_steps)

    assert torch.allclose(actions, expected)


def test_sample_actions_rs_returns_a_candidate_of_its_own_observation():
    """Guards the `repeat_interleave`/`view` index alignment: the action
    returned for observation `i` must be one of the `num_samples` candidates
    generated *for observation i*. A `.repeat()`-style (vs.
    `.repeat_interleave()`-style) expansion would produce the same shapes and
    stay in-bounds while silently mixing up which candidates belong to which
    observation -- shape/bounds-only assertions can't catch that."""
    agent = _make_agent(num_samples=4)
    policy = agent.policy
    batch_size, num_samples = 3, 4
    features = torch.randn(batch_size, policy.features_extractor.features_dim)

    torch.manual_seed(0)
    actions = policy.sample_actions_rs(
        features,
        features,
        num_samples=num_samples,
        clip_range=agent.return_clip_range,
        agg=agent.q_agg,
    )

    torch.manual_seed(0)
    noises = torch.randn(batch_size * num_samples, 3)
    candidates = policy.compute_flow_actions(
        features.repeat_interleave(num_samples, dim=0), noises, policy.num_flow_steps
    ).view(batch_size, num_samples, 3)

    for i in range(batch_size):
        assert (candidates[i] == actions[i]).all(dim=-1).any(), i


@pytest.mark.parametrize("policy_extraction", ["rs", "rpg"])
def test_predict_respects_policy_extraction(policy_extraction):
    agent = _make_agent(policy_extraction=policy_extraction, num_samples=3)
    assert agent.policy.policy_extraction == policy_extraction

    obs = torch.randn(4, *agent.env.single_observation_space.shape)
    with torch.no_grad():
        action = agent.policy.predict(obs)

    assert action.shape == (4, 3)
    assert torch.all(action >= agent.policy.action_low)
    assert torch.all(action <= agent.policy.action_high)
