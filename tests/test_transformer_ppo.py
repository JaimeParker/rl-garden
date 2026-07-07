from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import TransformerPPO
from rl_garden.encoders.base import BaseFeaturesExtractor


class StructuredFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal fake extractor declaring a token_and_prop layout, purely to
    exercise TransformerPPO's ViT opt-out at construction time."""

    def __init__(self, observation_space) -> None:
        super().__init__(observation_space, features_dim=16)

    def structured_feature_config(self):
        return {"layout": "token_and_prop", "num_patches": 4, "patch_dim": 4, "prop_dim": 0}

    def extract(self, obs, stop_gradient: bool = False) -> torch.Tensor:
        return torch.randn(obs.shape[0], 16)


class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
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
                "rgb": torch.randint(0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
                "state": torch.randn(self.num_envs, 4),
            }
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


class TransformerDoneVecEnv:
    """Like DummyVecEnv, but env 0 terminates at a chosen global step and reports
    ``final_observation``, exercising the bootstrap-value and window-boundary
    memory-reset paths that never fire with DummyVecEnv."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box, done_at_step: int) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(action_space.low, (self.num_envs,) + action_space.shape),
            high=np.broadcast_to(action_space.high, (self.num_envs,) + action_space.shape),
            dtype=action_space.dtype,
        )
        self._step = 0
        self._done_at_step = done_at_step

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        self._step += 1
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: dict = {}
        if self._step == self._done_at_step:
            terminations[0] = True
            infos["final_observation"] = self._obs()
        obs = self._obs()  # gymnasium autoreset convention: next obs is already reset
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        return None

    def _obs(self):
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


def _transformer_kwargs() -> dict[str, object]:
    # Deliberately tiny for fast tests -- shape correctness, not capacity.
    return {
        "embed_dim": 8,
        "head_dim": 4,
        "num_heads": 2,
        "num_transformer_layers": 1,
        "mlp_num": 1,
        "memory_len": 2,
    }


def _capture_train_losses(agent):
    captured: list[dict] = []
    original_train = agent.train

    def _wrapped():
        result = original_train()
        captured.append(result)
        return result

    agent.train = _wrapped
    return captured


def test_transformer_ppo_learn_one_iteration_state():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerPPO(env=env, **_ppo_kwargs(), **_transformer_kwargs())
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert agent._global_update == 1
    assert agent.policy.recurrent_encoder is not None
    assert len(captured) == 1
    assert torch.isfinite(torch.tensor(captured[0]["loss"]))
    assert torch.isfinite(torch.tensor(captured[0]["value_loss"]))


def test_transformer_ppo_handles_episode_termination_across_windows():
    env = TransformerDoneVecEnv(_state_space(), _action_space(), done_at_step=2)
    agent = TransformerPPO(env=env, **_ppo_kwargs(), **_transformer_kwargs())
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=8)  # 2 rollout windows of num_steps=2

    assert agent._global_step == 8
    assert agent._global_update == 2
    assert len(captured) == 2
    for losses in captured:
        assert torch.isfinite(torch.tensor(losses["loss"]))
        assert torch.isfinite(torch.tensor(losses["value_loss"]))
    assert torch.isfinite(agent.rollout_buffer.values).all()


def test_transformer_ppo_dict_obs_smoke():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = TransformerPPO(env=env, **_ppo_kwargs(), image_keys=("rgb",), **_transformer_kwargs())
    captured = _capture_train_losses(agent)

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert torch.isfinite(torch.tensor(captured[0]["loss"]))


def test_transformer_ppo_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerPPO(env=env, **_ppo_kwargs(), **_transformer_kwargs())
    agent.learn(total_timesteps=4)
    path = tmp_path / "transformer_ppo.pt"
    agent.save(path)

    loaded = TransformerPPO(
        env=DummyVecEnv(_state_space(), _action_space()),
        **_ppo_kwargs(),
        **_transformer_kwargs(),
    )
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_transformer_ppo_evaluate_with_eval_env_does_not_crash():
    """Regression test: periodic eval must use the stateful predict_recurrent
    path (shared SequencePPO eval hooks), not the stateless policy.predict()
    which would crash on the pre-encoder-vs-post-encoder feature-dim mismatch."""
    env = DummyVecEnv(_state_space(), _action_space())
    eval_env = DummyVecEnv(_state_space(), _action_space())
    agent = TransformerPPO(
        env=env, eval_env=eval_env, **_ppo_kwargs(), **_transformer_kwargs()
    )

    metrics = agent._evaluate()

    assert isinstance(metrics, dict)


def test_transformer_ppo_rejects_num_envs_not_divisible_by_minibatches():
    env = DummyVecEnv(_state_space(), _action_space())  # num_envs=2
    kwargs = _ppo_kwargs()
    kwargs["num_minibatches"] = 3
    with pytest.raises(ValueError):
        TransformerPPO(env=env, **kwargs, **_transformer_kwargs())


def test_transformer_ppo_rejects_structured_vit_features():
    env = DummyVecEnv(_state_space(), _action_space())
    with pytest.raises(NotImplementedError):
        TransformerPPO(
            env=env,
            policy_kwargs={"features_extractor_class": StructuredFeaturesExtractor},
            **_ppo_kwargs(),
            **_transformer_kwargs(),
        )
