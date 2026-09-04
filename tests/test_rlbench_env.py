"""Tests for the RLBench env backend (fake Environment/TaskEnvironment,
Sync/AsyncVectorEnv + torch adapter). Monkeypatches the entire ``rlbench``
package tree in ``sys.modules`` -- no real ``pyrep``/CoppeliaSim install
needed, same strategy as ``test_rlbench_dataset.py``.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.rlbench import RLBenchBackend
from rl_garden.envs.rlbench.config import RLBenchEnvConfig
from rl_garden.envs.rlbench.env import make_rlbench_env


class _FakeObservation:
    def __init__(self, step: int, images: dict | None = None):
        self._step = step
        self.joint_velocities = np.full(7, float(step), dtype=np.float32)
        self.gripper_open = 1.0
        self._images = images or {}

    def get_low_dim_data(self):
        return np.array([float(self._step)], dtype=np.float32)

    def __getattr__(self, name):
        if name in self._images:
            return self._images[name]
        raise AttributeError(name)


def _make_images(step: int) -> dict:
    images = {}
    for camera in ("left_shoulder", "right_shoulder", "overhead", "wrist", "front"):
        images[f"{camera}_rgb"] = np.full((4, 4, 3), step, dtype=np.uint8)
        images[f"{camera}_depth"] = np.full((4, 4), float(step), dtype=np.float32)
    return images


class _FakeTaskEnv:
    def __init__(self, with_images: bool, terminate_at: int = 5):
        self._with_images = with_images
        self._terminate_at = terminate_at
        self._t = 0
        self.last_reset_seed = None

    def _obs(self):
        return _FakeObservation(self._t, _make_images(self._t) if self._with_images else None)

    def reset(self, demo=None):
        self._t = 0
        return [], self._obs()

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._terminate_at
        return self._obs(), 1.0, terminated


class _FakeEnvironment:
    def __init__(self, *, action_mode, obs_config, headless, **env_kwargs):
        self.action_mode = action_mode
        self.obs_config = obs_config
        self.headless = headless
        self.env_kwargs = env_kwargs
        self.action_shape = (8,)
        self.launched = False
        self.shutdown_called = False
        self._task_env: _FakeTaskEnv | None = None

    def launch(self):
        self.launched = True

    def get_task(self, task_class):
        with_images = getattr(task_class, "_with_images", False)
        self._task_env = _FakeTaskEnv(with_images=with_images)
        return self._task_env

    def shutdown(self):
        self.shutdown_called = True


class _FakeMoveArmThenGripper:
    def __init__(self, arm_action_mode=None, gripper_action_mode=None):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    def action_bounds(self):
        return np.array([-1.0] * 8), np.array([1.0] * 8)


def _with_images_task_class():
    class _Task:
        _with_images = True

    return _Task


def _state_only_task_class():
    class _Task:
        _with_images = False

    return _Task


def _install_fake_rlbench(monkeypatch, *, task_class):
    fake_rlbench = types.ModuleType("rlbench")
    fake_utils = types.ModuleType("rlbench.utils")
    fake_observation_config = types.ModuleType("rlbench.observation_config")
    fake_environment = types.ModuleType("rlbench.environment")
    fake_action_modes = types.ModuleType("rlbench.action_modes")
    fake_action_mode = types.ModuleType("rlbench.action_modes.action_mode")
    fake_arm_action_modes = types.ModuleType("rlbench.action_modes.arm_action_modes")
    fake_gripper_action_modes = types.ModuleType("rlbench.action_modes.gripper_action_modes")

    fake_utils.name_to_task_class = lambda task_name: task_class
    fake_utils.get_stored_demos = lambda **kwargs: []

    class _FakeCameraConfig:
        def __init__(self):
            self.rgb = True
            self.depth = True
            self.point_cloud = True
            self.mask = True
            self.image_size = (128, 128)

        def set_all(self, value):
            self.rgb = value
            self.depth = value
            self.point_cloud = value
            self.mask = value

    class _FakeObservationConfig:
        def __init__(self):
            for camera in ("left_shoulder", "right_shoulder", "overhead", "wrist", "front"):
                setattr(self, f"{camera}_camera", _FakeCameraConfig())

        def set_all_low_dim(self, value):
            pass

        def set_all_high_dim(self, value):
            for camera in ("left_shoulder", "right_shoulder", "overhead", "wrist", "front"):
                getattr(self, f"{camera}_camera").set_all(value)

    fake_observation_config.ObservationConfig = _FakeObservationConfig
    fake_environment.Environment = _FakeEnvironment
    fake_action_mode.MoveArmThenGripper = _FakeMoveArmThenGripper
    fake_arm_action_modes.JointVelocity = lambda: None
    fake_gripper_action_modes.Discrete = lambda: None

    fake_rlbench.utils = fake_utils
    fake_rlbench.observation_config = fake_observation_config
    fake_rlbench.environment = fake_environment
    fake_rlbench.action_modes = fake_action_modes

    monkeypatch.setitem(sys.modules, "rlbench", fake_rlbench)
    monkeypatch.setitem(sys.modules, "rlbench.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "rlbench.observation_config", fake_observation_config)
    monkeypatch.setitem(sys.modules, "rlbench.environment", fake_environment)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes", fake_action_modes)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.action_mode", fake_action_mode)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.arm_action_modes", fake_arm_action_modes)
    monkeypatch.setitem(sys.modules, "rlbench.action_modes.gripper_action_modes", fake_gripper_action_modes)


def test_state_mode_returns_flat_box_torch_observation(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_state_only_task_class())

    cfg = RLBenchEnvConfig(task_name="reach_target", num_envs=2, seed=0, obs_mode="state", device="cpu")
    env = make_rlbench_env(cfg)

    obs, _ = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (2, 1)
    assert obs.dtype == torch.float32

    actions = torch.zeros(2, 8)
    next_obs, rewards, terminations, _truncations, _infos = env.step(actions)
    assert isinstance(next_obs, torch.Tensor)
    assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    assert isinstance(terminations, torch.Tensor) and terminations.dtype == torch.bool
    env.close()


def test_rgb_mode_returns_dict_with_renamed_image_keys(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_with_images_task_class())

    cfg = RLBenchEnvConfig(
        task_name="reach_target", num_envs=1, seed=0, obs_mode="rgb", cameras=("front",), device="cpu"
    )
    env = make_rlbench_env(cfg)

    obs, _ = env.reset()
    assert isinstance(obs, dict)
    assert set(obs) == {"state", "rgb_front", "depth_front"}
    assert obs["rgb_front"].dtype == torch.uint8
    assert obs["rgb_front"].shape == (1, 4, 4, 3)
    assert obs["depth_front"].shape == (1, 4, 4, 1)
    env.close()


def test_env_fn_forwards_env_kwargs(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_state_only_task_class())

    cfg = RLBenchEnvConfig(
        task_name="reach_target", num_envs=1, seed=0, env_kwargs={"static_positions": True}
    )
    env = make_rlbench_env(cfg)
    inner = env.unwrapped.envs[0].unwrapped
    assert inner._env.env_kwargs == {"static_positions": True}
    env.close()


def test_make_rlbench_env_selects_vector_backend_by_vectorization(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_state_only_task_class())
    calls = []
    import gymnasium as gym

    real_sync = gym.vector.SyncVectorEnv

    def fake_sync(env_fns, **kwargs):
        calls.append("sync")
        return real_sync(env_fns, **kwargs)

    def fake_async(env_fns, **kwargs):
        calls.append("async")
        return real_sync(env_fns, **kwargs)

    monkeypatch.setattr("gymnasium.vector.SyncVectorEnv", fake_sync)
    monkeypatch.setattr("gymnasium.vector.AsyncVectorEnv", fake_async)

    env_sync = make_rlbench_env(
        RLBenchEnvConfig(task_name="reach_target", num_envs=1, seed=0, vectorization="sync")
    )
    env_sync.close()
    env_async = make_rlbench_env(
        RLBenchEnvConfig(task_name="reach_target", num_envs=1, seed=0, vectorization="async")
    )
    env_async.close()

    assert calls == ["sync", "async"]


def test_seeded_vec_env_defaults_reset_to_configured_seed(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_state_only_task_class())

    env = make_rlbench_env(RLBenchEnvConfig(task_name="reach_target", num_envs=1, seed=42, device="cpu"))
    env.reset()
    env.reset(seed=7)
    env.close()


def test_reward_scale_bias_applied_when_non_trivial(monkeypatch):
    _install_fake_rlbench(monkeypatch, task_class=_state_only_task_class())

    env = make_rlbench_env(
        RLBenchEnvConfig(
            task_name="reach_target", num_envs=1, seed=0, reward_scale=2.0, reward_bias=1.0
        )
    )
    env.reset()
    _, rewards, _, _, _ = env.step(torch.zeros(1, 8))
    assert rewards.tolist() == [3.0]  # 1.0 * 2.0 + 1.0
    env.close()


def test_backend_make_train_env_uses_num_envs_and_train_config(monkeypatch):
    captured = {}

    def _fake_make_rlbench_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-train-env"

    monkeypatch.setattr("rl_garden.envs.rlbench.env.make_rlbench_env", _fake_make_rlbench_env)
    monkeypatch.setattr("rl_garden.envs.rlbench.make_rlbench_env", _fake_make_rlbench_env)

    req = EnvRequest(
        env_id="reach_target",
        num_envs=4,
        obs_mode="state",
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        camera_width=None,
        camera_height=None,
        num_eval_envs=2,
        backend_config=None,
    )
    result = RLBenchBackend.make_train_env(req)
    assert result == "sentinel-train-env"
    cfg = captured["cfg"]
    assert cfg.task_name == "reach_target"
    assert cfg.num_envs == 4
    assert cfg.vectorization == "sync"


def test_backend_make_eval_env_uses_num_eval_envs(monkeypatch):
    captured = {}

    def _fake_make_rlbench_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-eval-env"

    monkeypatch.setattr("rl_garden.envs.rlbench.env.make_rlbench_env", _fake_make_rlbench_env)
    monkeypatch.setattr("rl_garden.envs.rlbench.make_rlbench_env", _fake_make_rlbench_env)

    req = EnvRequest(
        env_id="reach_target",
        num_envs=4,
        obs_mode="rgb",
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        camera_width=None,
        camera_height=None,
        num_eval_envs=2,
        backend_config=None,
    )
    result = RLBenchBackend.make_eval_env(req)
    assert result == "sentinel-eval-env"
    assert captured["cfg"].num_envs == 2


def test_backend_resolve_config_parses_env_kwargs_json_and_vectorization():
    from rl_garden.common.env_args import RLBenchConfig

    req = EnvRequest(
        env_id="reach_target",
        num_envs=4,
        obs_mode="rgb",
        control_mode="",
        render_mode="rgb_array",
        seed=3,
        camera_width=None,
        camera_height=None,
        num_eval_envs=2,
        reward_scale=2.0,
        reward_bias=0.5,
        backend_config=RLBenchConfig(
            device="cpu", env_kwargs_json='{"static_positions": true}', vectorization="async"
        ),
    )
    cfg = RLBenchBackend.resolve_config(req, is_eval=False)
    assert cfg.task_name == "reach_target"
    assert cfg.env_kwargs == {"static_positions": True}
    assert cfg.vectorization == "async"
    assert cfg.reward_scale == 2.0
    assert cfg.reward_bias == 0.5
