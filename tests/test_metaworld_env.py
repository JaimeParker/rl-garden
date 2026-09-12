"""Tests for the Meta-World env backend. Monkeypatches ``gymnasium.make``/
``gymnasium.make_vec`` (and inserts an empty fake ``metaworld`` module into
``sys.modules``) rather than the real package -- no real MuJoCo/metaworld
install needed, same strategy as ``test_ogbench_dataset.py``'s fake-module
injection but at the ``gym.make`` boundary, since Meta-World's own env
construction goes through gymnasium's registry rather than a bespoke class
tree the way RLBench's does.
"""
from __future__ import annotations

import sys
import types

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.metaworld import MetaWorldBackend
from rl_garden.envs.metaworld.config import MetaWorldEnvConfig
from rl_garden.envs.metaworld.env import (
    _METAWORLD_CAMERAS,
    _MetaWorldEpisodeMetrics,
    _MetaWorldVisionWrapper,
    _validate_cameras,
    make_metaworld_env,
)
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


class _FakeSingleTaskEnv(gym.Env):
    """Minimal gymnasium.Env stand-in for what
    ``gym.make("Meta-World/MT1", ...)`` would return."""

    render_mode = None

    def __init__(self, terminate_at: int = 3):
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._t = 0
        self._terminate_at = terminate_at
        self.metadata: dict = {}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(3, dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._terminate_at
        obs = np.full(3, float(self._t), dtype=np.float64)
        info = {"success": 1.0 if terminated else 0.0}
        return obs, 1.0, terminated, False, info

    def close(self):
        pass


class _FakeVectorEnv(gym.vector.VectorEnv):
    """Minimal gymnasium.vector.VectorEnv stand-in for what
    ``gym.make_vec("Meta-World/MT10"|"MT50", ...)`` would return."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.metadata: dict = {}
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float64)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (num_envs, 2), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros((self.num_envs, 4), dtype=np.float64), {}

    def step(self, actions):
        obs = np.zeros((self.num_envs, 4), dtype=np.float64)
        reward = np.ones(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


def _install_fake_metaworld(monkeypatch):
    monkeypatch.setitem(sys.modules, "metaworld", types.ModuleType("metaworld"))


def test_single_task_env_id_builds_sync_vector_env_via_gym_make(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    captured_calls = []

    def fake_make(env_id, **kwargs):
        captured_calls.append((env_id, kwargs))
        return _FakeSingleTaskEnv()

    monkeypatch.setattr("gymnasium.make", fake_make)

    cfg = MetaWorldEnvConfig(env_id="reach-v3", num_envs=2, seed=5)
    env = make_metaworld_env(cfg)

    assert env.num_envs == 2
    assert [call[0] for call in captured_calls] == ["Meta-World/MT1", "Meta-World/MT1"]
    assert [call[1]["env_name"] for call in captured_calls] == ["reach-v3", "reach-v3"]
    assert [call[1]["seed"] for call in captured_calls] == [5, 6]

    obs, _ = env.reset()
    assert set(obs.keys()) == {"state"}
    assert isinstance(obs["state"], torch.Tensor)
    assert obs["state"].shape == (2, 3)
    env.close()


def test_multi_task_env_id_uses_gym_make_vec_and_ignores_num_envs(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    captured = {}

    def fake_make_vec(env_id, **kwargs):
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return _FakeVectorEnv(num_envs=10)

    monkeypatch.setattr("gymnasium.make_vec", fake_make_vec)

    cfg = MetaWorldEnvConfig(env_id="MT10", num_envs=999, seed=3, use_one_hot=True, vectorization="async")
    env = make_metaworld_env(cfg)

    assert captured["env_id"] == "Meta-World/MT10"
    assert captured["kwargs"] == {"vector_strategy": "async", "seed": 3, "use_one_hot": True}
    assert env.num_envs == 10  # not 999 -- upstream fixes the count, not this backend.
    env.close()


def test_vectorization_choice_sync_vs_async(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: _FakeSingleTaskEnv())

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

    env_sync = make_metaworld_env(MetaWorldEnvConfig(env_id="reach-v3", num_envs=1, seed=0, vectorization="sync"))
    env_sync.close()
    env_async = make_metaworld_env(
        MetaWorldEnvConfig(env_id="reach-v3", num_envs=1, seed=0, vectorization="async")
    )
    env_async.close()

    assert calls == ["sync", "async"]


def test_seeded_vec_env_defaults_reset_to_configured_seed(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: _FakeSingleTaskEnv())

    env = make_metaworld_env(MetaWorldEnvConfig(env_id="reach-v3", num_envs=1, seed=42))
    env.reset()
    env.reset(seed=7)
    env.close()


def test_reward_scale_bias_applied_when_non_trivial(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: _FakeSingleTaskEnv())

    env = make_metaworld_env(
        MetaWorldEnvConfig(env_id="reach-v3", num_envs=1, seed=0, reward_scale=2.0, reward_bias=1.0)
    )
    env.reset()
    _, rewards, _, _, _ = env.step(torch.zeros(1, 2))
    assert rewards.tolist() == [3.0]  # 1.0 * 2.0 + 1.0
    env.close()


class _FakeRecordEpisodeStatisticsEnv(gym.Env):
    """Stands in for a real Meta-World env after gymnasium's own
    ``RecordEpisodeStatistics`` has already applied -- ``info["episode"]``
    is present at episode end, alongside the raw ``info["success"]``
    Meta-World's ``AutoTerminateOnSuccessWrapper`` sets every step, same as
    the real wrapper stack ``gym.make("Meta-World/MT1", ...)`` produces."""

    render_mode = None

    def __init__(self, succeeds: bool):
        self._succeeds = succeeds
        self.metadata: dict = {}

    def step(self, action):
        info = {"success": 1.0 if self._succeeds else 0.0, "episode": {"r": 1.0, "l": 1}}
        return np.zeros(1), 1.0, True, False, info


def test_episode_metrics_wrapper_attaches_success_at_end_on_termination():
    env = _MetaWorldEpisodeMetrics(_FakeRecordEpisodeStatisticsEnv(succeeds=True))
    _, _, terminated, _, info = env.step(None)
    assert terminated
    assert info["episode"]["success_at_end"] == 1.0


def test_episode_metrics_wrapper_reports_zero_on_failed_episode():
    env = _MetaWorldEpisodeMetrics(_FakeRecordEpisodeStatisticsEnv(succeeds=False))
    _, _, _, _, info = env.step(None)
    assert info["episode"]["success_at_end"] == 0.0


class _FakeUnwrapped:
    def __init__(self):
        self.model = "fake-model"
        self.data = "fake-data"


class _FakeStateEnvForVision(gym.Env):
    """Minimal single-task stand-in with a real ``.unwrapped`` (needed by
    ``_MetaWorldVisionWrapper.__init__``, which reads
    ``env.unwrapped.model``/``.data``)."""

    render_mode = None

    def __init__(self):
        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.metadata: dict = {}
        self.unwrapped_ = _FakeUnwrapped()

    @property
    def unwrapped(self):
        return self.unwrapped_

    def reset(self, *, seed=None, options=None):
        return np.zeros(3, dtype=np.float64), {}

    def step(self, action):
        return np.ones(3, dtype=np.float64), 1.0, False, False, {}


class _FakeRenderer:
    instances: list["_FakeRenderer"] = []

    def __init__(self, model, data, *, width, height, camera_name):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.closed = False
        _FakeRenderer.instances.append(self)

    def render(self, mode):
        if mode == "rgb_array":
            return np.full((self.height, self.width, 3), 7, dtype=np.uint8)
        return np.full((self.height, self.width), 0.99, dtype=np.float32)

    def close(self):
        self.closed = True


def test_vision_wrapper_builds_dict_obs_space_and_renders_state_plus_rgbd(monkeypatch):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("rl_garden.envs.metaworld.env.MujocoRenderer", _FakeRenderer)

    env = _MetaWorldVisionWrapper(
        _FakeStateEnvForVision(),
        rgb_cameras=("corner2",),
        depth_cameras=("corner2",),
        image_size=(4, 4),
        state=True,
    )

    assert set(env.observation_space.spaces) == {"state", "rgb_corner2", "depth_corner2"}
    assert env.observation_space["rgb_corner2"].shape == (4, 4, 3)
    assert env.observation_space["depth_corner2"].shape == (4, 4, 1)

    obs, _ = env.reset()
    assert obs["state"].dtype == np.float32
    assert obs["rgb_corner2"].shape == (4, 4, 3)
    assert obs["depth_corner2"].shape == (4, 4, 1)

    obs, _, _, _, _ = env.step(np.zeros(2))
    assert obs["state"].tolist() == [1.0, 1.0, 1.0]
    assert obs["rgb_corner2"].dtype == np.uint8

    env.close()
    assert _FakeRenderer.instances[0].closed


def test_vision_wrapper_drops_state_key_when_state_false(monkeypatch):
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("rl_garden.envs.metaworld.env.MujocoRenderer", _FakeRenderer)

    env = _MetaWorldVisionWrapper(
        _FakeStateEnvForVision(),
        rgb_cameras=("corner2",),
        depth_cameras=(),
        image_size=(4, 4),
        state=False,
    )

    assert set(env.observation_space.spaces) == {"rgb_corner2"}
    obs, _ = env.reset()
    assert set(obs.keys()) == {"rgb_corner2"}
    env.close()


def test_vision_wrapper_rejects_unknown_camera(monkeypatch):
    monkeypatch.setattr("rl_garden.envs.metaworld.env.MujocoRenderer", _FakeRenderer)

    import pytest

    with pytest.raises(ObservationContractError, match="unknown camera"):
        _MetaWorldVisionWrapper(
            _FakeStateEnvForVision(),
            rgb_cameras=("not_a_real_camera",),
            depth_cameras=(),
            image_size=(4, 4),
            state=True,
        )


def test_validate_cameras_lists_available_cameras_on_unknown():
    import pytest

    with pytest.raises(ObservationContractError) as excinfo:
        _validate_cameras(("bogus",))
    for camera in _METAWORLD_CAMERAS:
        assert camera in str(excinfo.value)


def test_make_metaworld_env_rejects_vision_for_mt10(monkeypatch):
    _install_fake_metaworld(monkeypatch)

    import pytest

    with pytest.raises(ObservationContractError, match="MT10"):
        make_metaworld_env(
            MetaWorldEnvConfig(env_id="MT10", num_envs=10, seed=0, rgb_cameras=("corner2",))
        )


def test_backend_resolve_config_rejects_frame_stack_without_camera():
    import pytest

    from rl_garden.common.env_args import MetaWorldConfig

    req = EnvRequest(
        env_id="reach-v3",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=3,
        observation=ObservationConfig(frame_stack=3),
        num_eval_envs=2,
        backend_config=MetaWorldConfig(),
    )
    with pytest.raises(ObservationContractError, match="frame_stack"):
        MetaWorldBackend.resolve_config(req, is_eval=False)


def test_frame_stack_applies_image_frame_stack_wrapper(monkeypatch):
    _install_fake_metaworld(monkeypatch)
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("rl_garden.envs.metaworld.env.MujocoRenderer", _FakeRenderer)
    monkeypatch.setattr("gymnasium.make", lambda env_id, **kwargs: _FakeStateEnvForVision())

    cfg = MetaWorldEnvConfig(
        env_id="reach-v3",
        num_envs=1,
        seed=0,
        rgb_cameras=("corner2",),
        image_size=(4, 4),
        frame_stack=3,
    )
    env = make_metaworld_env(cfg)
    obs, _ = env.reset()
    assert obs["rgb_corner2"].shape == (1, 3, 4, 4, 3)
    env.close()


def test_backend_resolve_config_translation():
    from rl_garden.common.env_args import MetaWorldConfig

    req = EnvRequest(
        env_id="reach-v3",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=3,
        observation=ObservationConfig(rgb=("corner3",), image_size=(64, 64)),
        num_eval_envs=2,
        reward_scale=2.0,
        reward_bias=0.5,
        backend_config=MetaWorldConfig(device="cpu", vectorization="async", use_one_hot=False),
    )
    cfg = MetaWorldBackend.resolve_config(req, is_eval=False)
    assert cfg.env_id == "reach-v3"
    assert cfg.num_envs == 4
    assert cfg.vectorization == "async"
    assert cfg.use_one_hot is False
    assert cfg.reward_scale == 2.0
    assert cfg.reward_bias == 0.5
    assert cfg.rgb_cameras == ("corner3",)
    assert cfg.depth_cameras == ()
    assert cfg.state is True
    assert cfg.image_size == (64, 64)


def test_backend_make_train_env_uses_num_envs(monkeypatch):
    captured = {}

    def _fake_make_metaworld_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-train-env"

    monkeypatch.setattr("rl_garden.envs.metaworld.env.make_metaworld_env", _fake_make_metaworld_env)
    monkeypatch.setattr("rl_garden.envs.metaworld.make_metaworld_env", _fake_make_metaworld_env)

    req = EnvRequest(
        env_id="reach-v3",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    result = MetaWorldBackend.make_train_env(req)
    assert result == "sentinel-train-env"
    assert captured["cfg"].num_envs == 4


def test_backend_make_eval_env_uses_num_eval_envs(monkeypatch):
    captured = {}

    def _fake_make_metaworld_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-eval-env"

    monkeypatch.setattr("rl_garden.envs.metaworld.env.make_metaworld_env", _fake_make_metaworld_env)
    monkeypatch.setattr("rl_garden.envs.metaworld.make_metaworld_env", _fake_make_metaworld_env)

    req = EnvRequest(
        env_id="reach-v3",
        num_envs=4,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    result = MetaWorldBackend.make_eval_env(req)
    assert result == "sentinel-eval-env"
    assert captured["cfg"].num_envs == 2
