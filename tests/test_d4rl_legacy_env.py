from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.envs.d4rl_legacy.config import D4RLLegacyEnvConfig
from rl_garden.envs.d4rl_legacy.env import make_d4rl_legacy_env


class _FakeLegacyEnv:
    metadata: ClassVar[dict] = {}

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=2)
        self.steps = 0

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.steps = 0
        return np.zeros(3, dtype=np.float64)

    def step(self, action):
        self.steps += 1
        success = self.steps == 2
        return (
            np.full(3, self.steps, dtype=np.float64),
            float(success),
            success,
            {},
        )

    def render(self):
        return None

    def close(self):
        pass


class _FakeNormalizedActionEnv(_FakeLegacyEnv):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-0.3, 0.0], dtype=np.float32),
            high=np.array([0.5, 1.6], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action = None

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        return super().step(action)


def test_legacy_env_exposes_normalized_actions_without_rescaling(monkeypatch):
    legacy_env = _FakeNormalizedActionEnv()
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: legacy_env,
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="door-binary-v0", num_envs=1, device="cpu")
    )

    assert np.array_equal(env.single_action_space.low, np.full(2, -1.0))
    assert np.array_equal(env.single_action_space.high, np.full(2, 1.0))
    assert env.single_action_space.dtype == np.float32
    assert np.array_equal(env.action_space.low, np.full((1, 2), -1.0))
    assert np.array_equal(env.action_space.high, np.full((1, 2), 1.0))
    assert env.action_space.dtype == np.float32

    env.reset()
    normalized_action = torch.tensor([[-1.0, 0.75]], dtype=torch.float32)
    env.step(normalized_action)
    assert np.array_equal(legacy_env.last_action, normalized_action.numpy()[0])
    assert legacy_env.last_action.dtype == np.float32
    env.close()


def test_legacy_env_exposes_torch_vector_contract_and_success(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeLegacyEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="antmaze-test-v2", num_envs=1, device="cpu")
    )

    obs, _ = env.reset()
    assert obs.dtype == torch.float32
    action = torch.zeros((1, 1))
    _, reward1, terminated1, _, _ = env.step(action)
    _, reward2, terminated2, _, info2 = env.step(action)

    assert reward1.tolist() == [0.0]
    assert not terminated1.any()
    assert reward2.tolist() == [1.0]
    assert terminated2.tolist() == [True]
    assert info2["final_info"]["episode"]["success_at_end"].tolist() == [1.0]
    env.close()


def test_legacy_train_reward_transform_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeLegacyEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(
            env_id="antmaze-test-v2",
            num_envs=1,
            device="cpu",
            reward_scale=10.0,
            reward_bias=-5.0,
        )
    )
    env.reset()
    _, reward, _, _, _ = env.step(torch.zeros((1, 1)))
    assert reward.tolist() == [-5.0]
    env.close()


class _FakeBinaryEnv(_FakeLegacyEnv):
    def __init__(self):
        super().__init__()
        self.spec = SimpleNamespace(max_episode_steps=3)

    def step(self, action):
        self.steps += 1
        success = self.steps == 2
        return (
            np.full(3, self.steps, dtype=np.float64),
            0.0 if success else -1.0,
            False,
            {"goal_achieved": success},
        )


def test_binary_goal_achieved_terminates_episode(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeBinaryEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="pen-binary-v0", num_envs=1, device="cpu")
    )

    env.reset()
    env.step(torch.zeros((1, 1)))
    _, _, terminated, truncated, info = env.step(torch.zeros((1, 1)))

    assert terminated.tolist() == [True]
    assert truncated.tolist() == [False]
    assert info["final_info"]["episode"]["success_at_end"].tolist() == [1.0]
    env.close()


def test_kitchen_reports_completed_stages(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeLegacyEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="kitchen-partial-v0", num_envs=1, device="cpu")
    )

    env.reset()
    env.step(torch.zeros((1, 1)))
    _, _, _, _, info = env.step(torch.zeros((1, 1)))

    episode = info["final_info"]["episode"]
    assert episode["num_stages_solved"].tolist() == [1.0]
    assert episode["normalized_score"].tolist() == [25.0]
    env.close()


class _FakeStandardAdroitEnv(_FakeLegacyEnv):
    def get_normalized_score(self, value):
        return value / 10.0


def test_standard_adroit_reports_d4rl_normalized_score(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeStandardAdroitEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="door-human-v1", num_envs=1, device="cpu")
    )

    env.reset()
    env.step(torch.zeros((1, 1)))
    _, _, _, _, info = env.step(torch.zeros((1, 1)))

    assert info["final_info"]["episode"]["normalized_score"].tolist() == [10.0]
    env.close()


def test_locomotion_reports_d4rl_normalized_score(monkeypatch):
    monkeypatch.setattr(
        "rl_garden.envs.d4rl_legacy.env._make_legacy_env",
        lambda env_id: _FakeStandardAdroitEnv(),
    )
    env = make_d4rl_legacy_env(
        D4RLLegacyEnvConfig(env_id="halfcheetah-medium-v2", num_envs=1, device="cpu")
    )

    env.reset()
    env.step(torch.zeros((1, 1)))
    _, _, _, _, info = env.step(torch.zeros((1, 1)))

    assert info["final_info"]["episode"]["normalized_score"].tolist() == [10.0]
    env.close()
