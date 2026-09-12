"""Tests for the ``custom`` env backend template (PointReach-v0)."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.custom import CustomBackend
from rl_garden.envs.custom import CustomEnvConfig, make_custom_env
from rl_garden.envs.custom.point_reach_env import PointReachEnv
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


def test_point_reach_env_satisfies_gymnasium_api():
    env = PointReachEnv()
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict) and set(obs) == {"state"}
    assert obs["state"].shape == (6,)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert "distance" in info


def test_make_custom_env_returns_torch_tensors_on_configured_device():
    cfg = CustomEnvConfig(env_id="PointReach-v0", num_envs=3, seed=0, device="cpu")
    env = make_custom_env(cfg)

    assert env.num_envs == 3
    obs, _ = env.reset(seed=0)
    assert isinstance(obs["state"], torch.Tensor)
    assert obs["state"].shape == (3, 6)
    assert obs["state"].device.type == "cpu"

    actions = torch.as_tensor(env.action_space.sample())
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(next_obs["state"], torch.Tensor) and next_obs["state"].shape == (3, 6)
    assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    assert isinstance(terminations, torch.Tensor) and terminations.dtype == torch.bool
    assert isinstance(truncations, torch.Tensor) and truncations.dtype == torch.bool


def test_backend_registers_with_api_version_2():
    from rl_garden.envs.backend_registry import discover_env_backends, _REGISTRY

    discover_env_backends()
    assert "custom" in _REGISTRY
    assert _REGISTRY["custom"] is CustomBackend
    assert CustomBackend.api_version == 2
    assert CustomBackend.config_field == "custom"


def _make_req(**overrides):
    defaults = dict(
        env_id="PointReach-v0",
        num_envs=4,
        control_mode="n/a",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    defaults.update(overrides)
    return EnvRequest(**defaults)


def test_resolve_config_is_side_effect_free_and_splits_train_eval_num_envs():
    req = _make_req()

    train_cfg = CustomBackend.resolve_config(req, is_eval=False)
    assert train_cfg.num_envs == 4

    eval_cfg = CustomBackend.resolve_config(req, is_eval=True)
    assert eval_cfg.num_envs == 2


def test_make_train_and_eval_env_construct_working_vectorized_envs():
    req = _make_req(num_envs=2, num_eval_envs=2)

    train_env = CustomBackend.make_train_env(req)
    eval_env = CustomBackend.make_eval_env(req)

    assert train_env.num_envs == 2
    assert eval_env.num_envs == 2
    train_env.reset(seed=0)
    eval_env.reset(seed=0)


def test_resolve_config_rejects_vision_request():
    req = _make_req(observation=ObservationConfig(rgb=("cam",)))

    with pytest.raises(ObservationContractError):
        CustomBackend.resolve_config(req, is_eval=False)


class _FakeBackendConfig:
    pass


class _StubBrokenEnv:
    """Minimal stub whose observation space violates the contract."""

    def __init__(self):
        self.num_envs = 1
        # "bogus" is not in the strict state/rgb_<cam>/depth_<cam> vocabulary,
        # so validate_observation_space itself raises.
        self.single_observation_space = gym.spaces.Dict(
            {"bogus": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))}
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        return self.single_observation_space.sample(), {}

    def step(self, action):
        obs = self.single_observation_space.sample()
        return obs, 0.0, False, False, {}


class _BrokenFakeBackend:
    api_version = 2
    config_field = "_broken_fake_backend"

    @classmethod
    def resolve_config(cls, req, *, is_eval: bool):
        return _FakeBackendConfig()

    @classmethod
    def make_train_env(cls, req):
        return _StubBrokenEnv()

    @classmethod
    def make_eval_env(cls, req):
        return _StubBrokenEnv()


def test_registry_rejects_backend_returning_non_contract_space():
    from rl_garden.envs.backend_registry import _REGISTRY, make_training_envs, register_env_backend

    name = "_broken_fake_backend"
    if name not in _REGISTRY:
        register_env_backend(name, _BrokenFakeBackend)

    req = _make_req(observation=ObservationConfig(), create_eval_env=False)

    with pytest.raises(ObservationContractError):
        make_training_envs(name, req)
