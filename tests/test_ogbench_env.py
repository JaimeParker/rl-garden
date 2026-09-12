"""Tests for the OGBench env backend (Sync/AsyncVectorEnv + torch adapter)."""
import os
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.ogbench import OGBenchBackend
from rl_garden.envs.ogbench.config import OGBenchEnvConfig
from rl_garden.envs.ogbench.env import _make_env_fn, make_ogbench_env
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


class _TinyEnv(gym.Env):
    """Deterministic single-obs-dim env that terminates after ``terminate_at`` steps."""

    def __init__(self, terminate_at: int = 5):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.terminate_at = terminate_at
        self.last_reset_seed = None
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_reset_seed = seed
        self._t = 0
        return np.array([0.0], dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.terminate_at
        return np.array([float(self._t)], dtype=np.float64), 1.0, terminated, False, {}


def _install_fake_ogbench_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "ogbench", types.SimpleNamespace())


def _install_fake_gym_make(monkeypatch, captured):
    real_env = _TinyEnv()

    def _fake_make(env_id, **env_kwargs):
        captured["env_id"] = env_id
        captured["env_kwargs"] = env_kwargs
        return real_env

    monkeypatch.setattr("gymnasium.make", _fake_make)
    return real_env


def test_reset_and_step_return_torch_tensors_on_configured_device(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    _install_fake_gym_make(monkeypatch, {})

    cfg = OGBenchEnvConfig(env_id="antmaze-large-singletask-task1-v0", num_envs=2, seed=0, device="cpu")
    env = make_ogbench_env(cfg)

    obs, _ = env.reset()
    assert set(obs.keys()) == {"state"}
    assert isinstance(obs["state"], torch.Tensor)
    assert obs["state"].shape == (2, 1)
    assert obs["state"].dtype == torch.float32  # float64 downcast by TorchVectorEnvAdapter

    actions = torch.zeros(2, 1)
    next_obs, rewards, terminations, _truncations, _infos = env.step(actions)
    assert isinstance(next_obs["state"], torch.Tensor)
    assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    assert isinstance(terminations, torch.Tensor) and terminations.dtype == torch.bool
    env.close()


class _SuccessEnv(gym.Env):
    """Terminates at a fixed step, reporting ``info["success"]`` on every
    step the way every OGBench task family does (unconditionally, not just
    at termination)."""

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, terminate_at: int = 3):
        self.terminate_at = terminate_at
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.array([0.0], dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.terminate_at
        success = 1.0 if terminated else 0.0
        return np.array([float(self._t)], dtype=np.float64), 1.0, terminated, False, {"success": success}


def test_episode_metrics_attaches_success_at_end_from_info(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **env_kwargs: _SuccessEnv(terminate_at=3))

    env_fn = _make_env_fn("cube-single-singletask-v0", {}, None)
    env = env_fn()
    env.reset()

    for _ in range(2):
        _, _, terminated, _truncated, info = env.step(np.zeros(1, dtype=np.float32))
        assert not terminated
        assert "episode" not in info

    _, _, terminated, _truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated
    assert info["episode"]["success_at_end"] == np.float32(1.0)


def test_env_fn_registers_ogbench_and_defaults_mujoco_gl(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    captured = {}
    _install_fake_gym_make(monkeypatch, captured)
    monkeypatch.delenv("MUJOCO_GL", raising=False)

    env_fn = _make_env_fn("cube-single-singletask-v0", {}, None)
    env_fn()

    assert captured["env_id"] == "cube-single-singletask-v0"
    assert os.environ["MUJOCO_GL"] == "egl"


def test_env_fn_does_not_override_existing_mujoco_gl(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    _install_fake_gym_make(monkeypatch, {})
    monkeypatch.setenv("MUJOCO_GL", "osmesa")

    env_fn = _make_env_fn("cube-single-singletask-v0", {}, None)
    env_fn()

    assert os.environ["MUJOCO_GL"] == "osmesa"


def test_env_fn_forwards_env_kwargs(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    captured = {}
    _install_fake_gym_make(monkeypatch, captured)

    env_fn = _make_env_fn("cube-single-singletask-v0", {"max_episode_steps": 123}, None)
    env_fn()

    assert captured["env_kwargs"] == {"max_episode_steps": 123}


def test_env_fn_wraps_pixel_obs_as_dict_with_fixed_camera_key(monkeypatch):
    """visual-* env ids: OGBench has no real camera name, so the single
    pixel key is always "rgb_ogbench" (matches the offline dataset loader's
    fixed key, see OGBenchBackend.resolve_config)."""
    _install_fake_ogbench_module(monkeypatch)

    class _PixelEnv(gym.Env):
        observation_space = spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action):
            return np.zeros((4, 4, 3), dtype=np.uint8), 0.0, False, False, {}

    monkeypatch.setattr("gymnasium.make", lambda env_id, **env_kwargs: _PixelEnv())

    env_fn = _make_env_fn("visual-cube-single-singletask-v0", {}, "ogbench")
    env = env_fn()
    obs, _ = env.reset()
    assert set(obs.keys()) == {"rgb_ogbench"}
    assert obs["rgb_ogbench"].shape == (4, 4, 3)


def test_make_ogbench_env_selects_vector_backend_by_vectorization(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    _install_fake_gym_make(monkeypatch, {})
    calls = []
    real_sync = gym.vector.SyncVectorEnv

    def fake_sync(env_fns, **kwargs):
        calls.append("sync")
        return real_sync(env_fns, **kwargs)

    def fake_async(env_fns, **kwargs):
        calls.append("async")
        # Stand in for AsyncVectorEnv without spawning real subprocesses --
        # this test only asserts which vector-env class gets selected.
        return real_sync(env_fns, **kwargs)

    monkeypatch.setattr("gymnasium.vector.SyncVectorEnv", fake_sync)
    monkeypatch.setattr("gymnasium.vector.AsyncVectorEnv", fake_async)

    env_sync = make_ogbench_env(
        OGBenchEnvConfig(env_id="antmaze-large-singletask-task1-v0", num_envs=1, seed=0, vectorization="sync")
    )
    env_sync.close()
    env_async = make_ogbench_env(
        OGBenchEnvConfig(env_id="antmaze-large-singletask-task1-v0", num_envs=1, seed=0, vectorization="async")
    )
    env_async.close()

    assert calls == ["sync", "async"]


def test_seeded_vec_env_defaults_reset_to_configured_seed(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    real_env = _install_fake_gym_make(monkeypatch, {})

    env = make_ogbench_env(
        OGBenchEnvConfig(env_id="antmaze-large-singletask-task1-v0", num_envs=1, seed=42, device="cpu")
    )
    env.reset()  # no seed passed -- should default to cfg.seed
    assert real_env.last_reset_seed == 42

    env.reset(seed=7)  # explicit seed wins over the configured default
    assert real_env.last_reset_seed == 7
    env.close()


def test_reward_scale_bias_applied_when_non_trivial(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)
    _install_fake_gym_make(monkeypatch, {})

    env = make_ogbench_env(
        OGBenchEnvConfig(
            env_id="antmaze-large-singletask-task1-v0",
            num_envs=1,
            seed=0,
            reward_scale=2.0,
            reward_bias=1.0,
        )
    )
    env.reset()
    _, rewards, _, _, _ = env.step(torch.zeros(1, 1))
    assert rewards.tolist() == [3.0]  # 1.0 * 2.0 + 1.0
    env.close()


def test_frame_stack_applies_image_frame_stack_wrapper_on_fixed_camera_key(monkeypatch):
    _install_fake_ogbench_module(monkeypatch)

    class _PixelEnv(gym.Env):
        observation_space = spaces.Box(low=0, high=255, shape=(4, 4, 3), dtype=np.uint8)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action):
            return np.zeros((4, 4, 3), dtype=np.uint8), 0.0, False, False, {}

    monkeypatch.setattr("gymnasium.make", lambda env_id, **env_kwargs: _PixelEnv())

    env = make_ogbench_env(
        OGBenchEnvConfig(
            env_id="visual-cube-single-singletask-v0",
            num_envs=2,
            seed=0,
            pixel_camera="ogbench",
            frame_stack=3,
        )
    )
    obs, _ = env.reset()
    assert obs["rgb_ogbench"].shape == (2, 3, 4, 4, 3)
    env.close()


def test_backend_make_train_env_uses_num_envs_and_train_config(monkeypatch):
    captured = {}

    def _fake_make_ogbench_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-train-env"

    monkeypatch.setattr("rl_garden.envs.ogbench.env.make_ogbench_env", _fake_make_ogbench_env)
    monkeypatch.setattr("rl_garden.envs.ogbench.make_ogbench_env", _fake_make_ogbench_env)

    req = EnvRequest(
        env_id="cube-single-singletask-v0",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    result = OGBenchBackend.make_train_env(req)
    assert result == "sentinel-train-env"
    cfg = captured["cfg"]
    assert cfg.env_id == "cube-single-singletask-v0"
    assert cfg.num_envs == 4
    assert cfg.vectorization == "sync"
    assert cfg.pixel_camera is None


def test_backend_make_eval_env_uses_num_eval_envs(monkeypatch):
    captured = {}

    def _fake_make_ogbench_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-eval-env"

    monkeypatch.setattr("rl_garden.envs.ogbench.env.make_ogbench_env", _fake_make_ogbench_env)
    monkeypatch.setattr("rl_garden.envs.ogbench.make_ogbench_env", _fake_make_ogbench_env)

    req = EnvRequest(
        env_id="visual-antmaze-medium-singletask-v0",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(rgb=("ogbench",), state=False),
        num_eval_envs=2,
        backend_config=None,
    )
    result = OGBenchBackend.make_eval_env(req)
    assert result == "sentinel-eval-env"
    cfg = captured["cfg"]
    assert cfg.num_envs == 2
    assert cfg.pixel_camera == "ogbench"


def test_backend_resolve_config_parses_env_kwargs_json_and_vectorization():
    from rl_garden.common.env_args import OGBenchConfig

    req = EnvRequest(
        env_id="cube-single-singletask-v0",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=3,
        observation=ObservationConfig(),
        num_eval_envs=2,
        reward_scale=2.0,
        reward_bias=0.5,
        backend_config=OGBenchConfig(
            device="cpu", env_kwargs_json='{"max_episode_steps": 50}', vectorization="async"
        ),
    )
    cfg = OGBenchBackend.resolve_config(req, is_eval=False)
    assert cfg.env_id == "cube-single-singletask-v0"
    assert cfg.env_kwargs == {"max_episode_steps": 50}
    assert cfg.vectorization == "async"
    assert cfg.reward_scale == 2.0
    assert cfg.reward_bias == 0.5


def _base_req(**overrides):
    defaults = dict(
        env_id="cube-single-singletask-v0",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    defaults.update(overrides)
    return EnvRequest(**defaults)


def test_resolve_config_requires_fixed_camera_name_for_visual_env_id():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("main",), state=False),
    )
    with pytest.raises(ObservationContractError, match="rgb=\\('ogbench',\\)"):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_multiple_cameras_for_visual_env_id():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench", "extra"), state=False),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_depth_for_visual_env_id():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench",), depth=("ogbench",), state=False),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_state_for_visual_env_id():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench",), state=True),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_rgb_for_non_visual_env_id():
    req = _base_req(
        env_id="cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench",), state=False),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_depth_for_non_visual_env_id():
    req = _base_req(
        env_id="cube-single-singletask-v0",
        observation=ObservationConfig(depth=("ogbench",)),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_frame_stack_for_non_visual_env_id():
    req = _base_req(
        env_id="cube-single-singletask-v0",
        observation=ObservationConfig(frame_stack=3),
    )
    with pytest.raises(ObservationContractError, match="frame_stack"):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_image_size_override():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench",), state=False, image_size=(32, 32)),
    )
    with pytest.raises(ObservationContractError):
        OGBenchBackend.resolve_config(req, is_eval=False)


def test_resolve_config_accepts_valid_visual_observation():
    req = _base_req(
        env_id="visual-cube-single-singletask-v0",
        observation=ObservationConfig(rgb=("ogbench",), state=False),
    )
    cfg = OGBenchBackend.resolve_config(req, is_eval=False)
    assert cfg.pixel_camera == "ogbench"


def test_resolve_config_accepts_valid_state_observation():
    req = _base_req(observation=ObservationConfig())
    cfg = OGBenchBackend.resolve_config(req, is_eval=False)
    assert cfg.pixel_camera is None
