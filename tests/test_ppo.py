from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import PPO
from rl_garden.buffers import RolloutBuffer
from rl_garden.encoders import FlattenExtractor
from rl_garden.policies.ppo_policy import PPOPolicy, get_ppo_arch


class DummyVecEnv:
    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box
    ) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(
                action_space.low, (self.num_envs,) + action_space.shape
            ),
            high=np.broadcast_to(
                action_space.high, (self.num_envs,) + action_space.shape
            ),
            dtype=action_space.dtype,
        )
        self._step = 0

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)
        self._step += 1
        obs = self._obs()
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def _obs(self):
        if isinstance(self.single_observation_space, spaces.Dict):
            return {
                "rgb": torch.randint(
                    0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8
                ),
                "state": torch.randn(self.num_envs, 4),
            }
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


def _state_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _dict_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _ppo_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "target_kl": None,
        "net_arch": [16],
    }


def test_get_ppo_arch_from_list_and_dict():
    assert get_ppo_arch([32, 16]) == ([32, 16], [32, 16])
    assert get_ppo_arch({"pi": [16], "vf": [32]}) == ([16], [32])


def test_ppo_policy_action_value_log_prob_shapes():
    obs_space = _state_space()
    action_space = _action_space()
    policy = PPOPolicy(
        obs_space,
        action_space,
        FlattenExtractor(obs_space),
        net_arch=[16],
    )
    obs = torch.randn(5, 4)
    actions, values, log_prob, entropy = policy(obs)

    assert actions.shape == (5, 2)
    assert values.shape == (5, 1)
    assert log_prob.shape == (5, 1)
    assert entropy.shape == (5, 1)

    values_eval, log_prob_eval, entropy_eval = policy.evaluate_actions(obs, actions)
    assert values_eval.shape == (5, 1)
    assert log_prob_eval.shape == (5, 1)
    assert entropy_eval.shape == (5, 1)


def test_rollout_buffer_computes_gae_returns():
    buffer = RolloutBuffer(
        _state_space(),
        _action_space(),
        num_steps=3,
        num_envs=2,
        device="cpu",
        gamma=1.0,
        gae_lambda=1.0,
    )
    for _ in range(3):
        buffer.add(
            torch.zeros(2, 4),
            torch.zeros(2, 2),
            torch.ones(2),
            torch.zeros(2),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        )

    buffer.compute_returns_and_advantage(torch.zeros(2), torch.zeros(2))

    assert torch.allclose(buffer.returns[0], torch.full((2,), 3.0))
    assert torch.allclose(buffer.returns[1], torch.full((2,), 2.0))
    assert torch.allclose(buffer.returns[2], torch.full((2,), 1.0))
    sample = next(buffer.get(batch_size=2))
    assert sample.obs.shape == (2, 4)
    assert sample.actions.shape == (2, 2)


def test_ppo_learn_one_iteration_state():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs())

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert agent._global_update == 1


def test_ppo_dict_obs_constructs_and_trains_one_update():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs(), image_keys=("rgb",))

    agent.learn(total_timesteps=4)

    assert agent.detach_encoder_on_actor is True
    assert agent._global_step == 4


def test_ppo_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs())
    agent.learn(total_timesteps=4)
    path = tmp_path / "ppo.pt"
    agent.save(path)

    loaded = PPO(env=DummyVecEnv(_state_space(), _action_space()), **_ppo_kwargs())
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    assert loaded._global_update == agent._global_update
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_ppo_value_extra_obs_keys_builds_separate_value_encoder():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "privileged": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )
    env = DummyVecEnv(obs_space, _action_space())
    agent = PPO(
        env=env,
        **_ppo_kwargs(),
        image_keys=("rgb",),
        value_extra_obs_keys=("privileged",),
    )
    assert agent.policy.value_features_extractor is not None
    assert "privileged" not in agent.policy.features_extractor._observation_space.spaces
    assert "privileged" in agent.policy.value_features_extractor._observation_space.spaces


def test_ppo_state_obs_rejects_value_kwargs():
    import pytest

    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(ValueError, match="image-related kwargs"):
        PPO(env=env, **_ppo_kwargs(), value_extra_obs_keys=("x",))
