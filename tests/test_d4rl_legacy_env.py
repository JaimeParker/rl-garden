from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.envs.d4rl_legacy.config import D4RLLegacyEnvConfig
from rl_garden.envs.d4rl_legacy.env import make_d4rl_legacy_env


class _FakeLegacyEnv:
    metadata = {}

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
