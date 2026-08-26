"""Tests for FlowBC: pure conditional flow-matching behavioral cloning."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import FlowBC, OfflineEnvSpec
from rl_garden.encoders.combined import default_image_encoder_factory
from rl_garden.policies.flow_bc_policy import FlowBCPolicy

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


def _make_agent(**kwargs) -> FlowBC:
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
    return FlowBC(**defaults)


def _fill(agent: FlowBC, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _fill_vision(agent: FlowBC, steps: int = 64) -> None:
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


def test_policy_is_flow_bc_policy_with_actor_vector_field():
    agent = _make_agent()
    assert isinstance(agent.policy, FlowBCPolicy)
    assert agent.policy.actor_bc_flow.use_time_conditioning is True


def test_loss_decreases_over_gradient_steps():
    # Flow-matching regression targets a fresh random x_0 every batch, so
    # per-step loss is noisy -- compare averaged early vs. late windows
    # instead of first-vs-last single steps.
    torch.manual_seed(0)
    agent = _make_agent()
    _fill(agent, steps=64)
    early = [agent.train(gradient_steps=1)["loss"] for _ in range(10)]
    for _ in range(60):
        agent.train(gradient_steps=1)
    late = [agent.train(gradient_steps=1)["loss"] for _ in range(10)]
    assert sum(late) / len(late) < sum(early) / len(early)


def test_predict_respects_action_bounds():
    agent = _make_agent()
    obs = torch.randn(8, 6)
    action = agent.policy.predict(obs)
    low = torch.as_tensor(agent.env.single_action_space.low)
    high = torch.as_tensor(agent.env.single_action_space.high)
    assert action.shape == (8, 3)
    assert torch.all(action >= low - 1e-5)
    assert torch.all(action <= high + 1e-5)


def test_vision_obs_construction_and_predict_shape():
    agent = _make_agent(
        env=_vision_env(),
        image_encoder_factory=_test_image_encoder_factory,
        image_keys=("rgb",),
        state_key="state",
    )
    _fill_vision(agent, steps=16)
    metrics = agent.train(gradient_steps=1)
    assert np.isfinite(metrics["loss"])

    obs = {
        "rgb": torch.randint(0, 256, (4, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=torch.uint8),
        "state": torch.randn(4, 6),
    }
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)


def test_checkpoint_round_trips_actor_bc_flow():
    agent = _make_agent()
    _fill(agent, steps=32)
    agent.train(gradient_steps=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "flow_bc.pt")
        agent.save(path)

        reloaded = _make_agent()
        reloaded.load(path, load_replay_buffer=False)

    for p1, p2 in zip(
        agent.policy.actor_bc_flow.parameters(),
        reloaded.policy.actor_bc_flow.parameters(),
    ):
        assert torch.equal(p1, p2)
