from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SACFlow
from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.networks import FlowMatchingActor
from rl_garden.policies.sac_flow_policy import SACFlowPolicy


class DummyVecEnv:
    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box, num_envs: int = 2
    ) -> None:
        self.num_envs = num_envs
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )

    def reset(self, seed: int | None = None):
        del seed
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0 + 1e-4)
        assert torch.all(actions >= -1.0 - 1e-4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def _obs(self):
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


class StructuredFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal fake extractor declaring a token_and_prop layout, purely to
    exercise SACFlow's structured-obs rejection at construction time."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=16)

    def structured_feature_config(self):
        return {"layout": "token_and_prop", "num_patches": 4, "patch_dim": 4, "prop_dim": 0}

    def extract(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        return torch.randn(obs.shape[0], 16)


def _state_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _sac_flow_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 32,
        "batch_size": 4,
        "learning_starts": 10,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [16],
        "flow_hidden_dims": [16],
        "denoising_steps": 2,
    }


def test_sac_flow_learn_one_iteration():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = SACFlow(env=env, **_sac_flow_kwargs())

    agent.learn(total_timesteps=40)

    assert agent._global_step == 40


def test_sac_flow_uses_flow_matching_actor():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = SACFlow(env=env, **_sac_flow_kwargs())

    assert isinstance(agent.policy, SACFlowPolicy)
    assert isinstance(agent.policy.actor, FlowMatchingActor)
    assert agent.policy.actor.denoising_steps == 2


def test_sac_flow_rejects_token_and_prop_features():
    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(NotImplementedError):
        SACFlow(
            env=env,
            policy_kwargs={"features_extractor_class": StructuredFeaturesExtractor},
            **_sac_flow_kwargs(),
        )


def test_sac_flow_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = SACFlow(env=env, **_sac_flow_kwargs())
    agent.learn(total_timesteps=40)
    path = tmp_path / "sac_flow.pt"
    agent.save(path)

    loaded = SACFlow(env=DummyVecEnv(_state_space(), _action_space()), **_sac_flow_kwargs())
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_flow_matching_actor_action_log_prob_finite_and_differentiable():
    torch.manual_seed(0)
    actor = FlowMatchingActor(
        features_dim=6,
        action_space=_action_space(),
        hidden_dims=[16, 16],
        denoising_steps=3,
        noise_std=0.3,
    )
    features = torch.randn(8, 6)

    action, log_prob = actor.action_log_prob(features)

    assert action.shape == (8, 2)
    assert log_prob.shape == (8, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.all(action <= 1.0 + 1e-5)
    assert torch.all(action >= -1.0 - 1e-5)

    loss = (action.sum() + log_prob.sum())
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_flow_matching_actor_log_prob_alone_reaches_velocity_network():
    """log_prob's only theta-dependent term is the final tanh-Jacobian
    correction (the per-step Gaussian transition terms are analytically
    constant in theta -- see the comment in flow_actor.py). Backprop
    log_prob alone (not summed with action) to catch a future refactor that
    accidentally detaches x before the tanh, which would silently zero the
    entropy term's gradient while every action-path test stays green."""
    torch.manual_seed(0)
    actor = FlowMatchingActor(
        features_dim=6,
        action_space=_action_space(),
        hidden_dims=[16, 16],
        denoising_steps=3,
        noise_std=0.3,
    )
    features = torch.randn(8, 6)

    _, log_prob = actor.action_log_prob(features)
    log_prob.sum().backward()

    assert actor.fc_velocity.weight.grad is not None
    assert torch.any(actor.fc_velocity.weight.grad != 0)


def test_flow_matching_actor_deterministic_action_matches_shape():
    actor = FlowMatchingActor(
        features_dim=6,
        action_space=_action_space(),
        hidden_dims=[16],
        denoising_steps=2,
        noise_std=0.3,
    )
    features = torch.randn(5, 6)

    action = actor.deterministic_action(features)

    assert action.shape == (5, 2)
    assert torch.isfinite(action).all()
