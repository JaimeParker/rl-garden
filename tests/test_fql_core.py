from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from rl_garden.algorithms import FQL, OfflineEnvSpec
from rl_garden.encoders.combined import CombinedExtractor, default_image_encoder_factory
from rl_garden.encoders.flatten import FlattenExtractor
from rl_garden.policies.fql_policy import FQLPolicy

# Small + fast: "gap" pooling (unlike the default "flatten") tolerates tiny
# images without PlainConv's flatten-layer size mismatch.
_TEST_IMAGE_SIZE = 16
_test_image_encoder_factory = default_image_encoder_factory(
    features_dim=16, plain_conv_pooling="gap"
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


def _make_agent(**kwargs) -> FQL:
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
    return FQL(**defaults)


def _fill(agent: FQL, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: FQL, steps: int = 64) -> None:
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


def _assert_predict_in_bounds(agent: FQL) -> None:
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
        FQL(env=unsupported, buffer_device="cpu", device="cpu")


def test_vision_shared_encoder_smoke():
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="shared",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "actor_loss", "bc_flow_loss", "distill_loss", "q_loss"):
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
    for key in ("critic_loss", "actor_loss", "bc_flow_loss", "distill_loss", "q_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])
    _assert_predict_in_bounds(agent)


def test_separate_encoder_produces_three_independent_instances():
    obs_space = _vision_env().single_observation_space
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _make_extractor():
        return CombinedExtractor(
            observation_space=obs_space, image_encoder_factory=_test_image_encoder_factory
        )

    shared_fe = _make_extractor()
    shared_policy = FQLPolicy(
        obs_space, act_space, shared_fe, net_arch=[16, 16], encoder_sharing="shared"
    )

    critic_fe = _make_extractor()
    bc_fe = _make_extractor()
    onestep_fe = _make_extractor()
    separate_policy = FQLPolicy(
        obs_space,
        act_space,
        critic_fe,
        net_arch=[16, 16],
        encoder_sharing="separate",
        actor_bc_flow_encoder=bc_fe,
        actor_onestep_flow_encoder=onestep_fe,
    )

    shared_encoder_params = sum(p.numel() for p in shared_policy.features_extractor.parameters())
    separate_encoder_params = (
        sum(p.numel() for p in separate_policy.features_extractor.parameters())
        + sum(p.numel() for p in separate_policy.actor_bc_flow_encoder.parameters())
        + sum(p.numel() for p in separate_policy.actor_onestep_flow_encoder.parameters())
    )
    assert separate_encoder_params == 3 * shared_encoder_params

    ptrs = set()
    for encoder in (
        separate_policy.features_extractor,
        separate_policy.actor_bc_flow_encoder,
        separate_policy.actor_onestep_flow_encoder,
    ):
        for p in encoder.parameters():
            assert p.data_ptr() not in ptrs, "encoder instances must not share storage"
            ptrs.add(p.data_ptr())


def test_separate_mode_actor_optimizer_excludes_critic_encoder():
    """The load-bearing isolation mechanism for encoder_sharing='separate':
    no torch.no_grad()/detach is used (unlike the shared-mode teacher-target
    case) -- isolation comes entirely from the critic's own encoder never
    appearing in actor_optimizer's parameter list."""
    agent = _make_agent(
        env=_vision_env(),
        encoder_sharing="separate",
        image_encoder_factory=_test_image_encoder_factory,
    )
    _fill_vision(agent)

    critic_encoder_ptrs = {p.data_ptr() for p in agent.policy.features_extractor.parameters()}
    actor_param_ptrs = {p.data_ptr() for p in agent.policy.actor_parameters()}
    assert critic_encoder_ptrs.isdisjoint(actor_param_ptrs)

    actor_optimizer_ptrs = {
        p.data_ptr() for group in agent.actor_optimizer.param_groups for p in group["params"]
    }
    assert critic_encoder_ptrs.isdisjoint(actor_optimizer_ptrs)


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    for key in ("critic_loss", "actor_loss", "bc_flow_loss", "distill_loss", "q_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_actor_and_critic_update_every_step_no_delay():
    """Unlike TD3-BC, FQL has no policy_freq-style delayed actor update --
    the reference backprops critic_loss and actor_loss together every step."""
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


def test_distill_loss_target_is_detached_from_teacher():
    """The one place the port needs an explicit torch.no_grad(): the
    teacher's Euler-unroll target for distill_loss must not backprop into
    actor_bc_flow. bc_flow_loss (a separate term) still must reach it."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    fe = FlattenExtractor(observation_space=obs_space)
    policy = FQLPolicy(obs_space, act_space, fe, net_arch=[16, 16])

    features = torch.randn(8, fe.features_dim)
    noises = torch.randn(8, 3)

    with torch.no_grad():
        target = policy.compute_flow_actions(features, noises, num_steps=4)
    actor_actions = policy.actor_onestep_flow(features, noises)
    distill_loss = F.mse_loss(actor_actions, target)

    policy.zero_grad()
    distill_loss.backward()
    bc_flow_grads = [p.grad for p in policy.actor_bc_flow.parameters()]
    onestep_grads = [p.grad for p in policy.actor_onestep_flow.parameters()]

    assert all(g is None or torch.all(g == 0) for g in bc_flow_grads)
    assert any(g is not None and torch.any(g != 0) for g in onestep_grads)


def test_train_computes_distill_target_without_grad(monkeypatch):
    """Regression guard on the production call site in FQLCore.train():
    accidentally dropping the torch.no_grad() around compute_flow_actions
    must fail this test, not just a standalone no_grad() written by hand."""
    agent = _make_agent()
    _fill(agent)

    seen_grad_enabled = []
    original = agent.policy.compute_flow_actions

    def spy(*args, **kwargs):
        seen_grad_enabled.append(torch.is_grad_enabled())
        return original(*args, **kwargs)

    monkeypatch.setattr(agent.policy, "compute_flow_actions", spy)
    agent.train(1)

    assert seen_grad_enabled
    assert not any(seen_grad_enabled)


def test_q_agg_min_is_not_silently_ignored():
    """q_agg='min' must actually change the critic target, not collapse to
    rl-garden's usual 'mean'/'min' default silently."""
    agent = _make_agent(q_agg="mean")
    q_all = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
    assert torch.equal(agent._aggregate_target_q(q_all), q_all.mean(dim=0))

    agent.q_agg = "min"
    assert torch.equal(agent._aggregate_target_q(q_all), q_all.min(dim=0).values)


def test_normalize_q_loss_scales_q_loss():
    torch.manual_seed(0)
    agent_norm = _make_agent(normalize_q_loss=True)
    _fill(agent_norm)
    torch.manual_seed(0)
    agent_plain = _make_agent(normalize_q_loss=False)
    _fill(agent_plain)

    metrics_norm = agent_norm.train(1, compute_info=True)
    metrics_plain = agent_plain.train(1, compute_info=True)
    assert metrics_norm["q_loss"] != pytest.approx(metrics_plain["q_loss"])


def test_bc_flow_loss_reaches_teacher_network():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    fe = FlattenExtractor(observation_space=obs_space)
    policy = FQLPolicy(obs_space, act_space, fe, net_arch=[16, 16])

    features = torch.randn(8, fe.features_dim)
    actions = torch.rand(8, 3) * 2 - 1
    x_0 = torch.randn(8, 3)
    t = torch.rand(8, 1)
    x_t = (1 - t) * x_0 + t * actions
    vel_target = actions - x_0

    pred_vel = policy.actor_bc_flow(features, x_t, t)
    bc_flow_loss = F.mse_loss(pred_vel, vel_target)

    policy.zero_grad()
    bc_flow_loss.backward()
    bc_flow_grads = [p.grad for p in policy.actor_bc_flow.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in bc_flow_grads)
