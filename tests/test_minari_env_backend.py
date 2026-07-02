"""Tests for the Minari env backend (SyncVectorEnv + torch adapter)."""
import sys
import types

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.minari import MinariBackend
from rl_garden.envs.minari.config import MinariEnvConfig
from rl_garden.envs.minari.env import make_minari_env


class _TinyEnv(gym.Env):
    """Deterministic single-obs-dim env that terminates after ``terminate_at`` steps."""

    def __init__(self, terminate_at: int, obs_start: float = 0.0):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.terminate_at = terminate_at
        self.obs_start = obs_start
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.array([self.obs_start], dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.array([self.obs_start + self._t], dtype=np.float32)
        terminated = self._t >= self.terminate_at
        return obs, 1.0, terminated, False, {}


class _FakeMinariDataset:
    def __init__(self, terminate_ats):
        self._terminate_ats = list(terminate_ats)
        self._call_idx = 0

    def recover_environment(self, eval_env: bool = False):
        terminate_at = self._terminate_ats[self._call_idx % len(self._terminate_ats)]
        self._call_idx += 1
        return _TinyEnv(terminate_at=terminate_at)


def _install_fake_minari(monkeypatch, dataset) -> None:
    fake_module = types.SimpleNamespace(load_dataset=lambda dataset_id, download=True: dataset)
    monkeypatch.setitem(sys.modules, "minari", fake_module)


def test_reset_and_step_return_torch_tensors_on_configured_device(monkeypatch):
    _install_fake_minari(monkeypatch, _FakeMinariDataset([5]))
    cfg = MinariEnvConfig(dataset_id="fake/dataset-v0", num_envs=3, eval_env=False, device="cpu")
    env = make_minari_env(cfg)

    obs, _ = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (3, 1)
    assert obs.device.type == "cpu"

    actions = torch.zeros(3, 1)
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(next_obs, torch.Tensor) and next_obs.shape == (3, 1)
    assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    assert isinstance(terminations, torch.Tensor) and terminations.dtype == torch.bool
    assert isinstance(truncations, torch.Tensor) and truncations.dtype == torch.bool
    assert env.num_envs == 3


def test_same_step_autoreset_final_observation_matches_pre_reset_obs(monkeypatch):
    _install_fake_minari(monkeypatch, _FakeMinariDataset([2]))
    cfg = MinariEnvConfig(dataset_id="fake/dataset-v0", num_envs=2, eval_env=False, device="cpu")
    env = make_minari_env(cfg)
    env.reset()

    actions = torch.zeros(2, 1)
    env.step(actions)  # step 1: not terminal yet
    next_obs, rewards, terminations, truncations, infos = env.step(actions)  # step 2: both terminate

    # Both envs terminate in lockstep (deterministic terminate_at=2).
    assert torch.equal(terminations, torch.tensor([True, True]))
    # SAME_STEP autoreset: the returned obs is already the reset (post-autoreset)
    # observation, not the true terminal one.
    assert torch.equal(next_obs.flatten(), torch.tensor([0.0, 0.0]))
    # The true pre-reset terminal observation (obs_start + 2 == 2.0) must be
    # recoverable from infos["final_observation"].
    assert "final_observation" in infos
    assert torch.equal(infos["final_observation"].flatten(), torch.tensor([2.0, 2.0]))
    # final_info/episode stats must be present with the naming off_policy.py expects.
    assert "final_info" in infos
    assert "episode" in infos["final_info"]
    assert torch.equal(infos["_final_info"], torch.tensor([True, True]))
    assert torch.equal(infos["final_info"]["episode"]["l"], torch.tensor([2, 2]))


def test_partial_termination_final_info_mask_is_per_env(monkeypatch):
    # env 0 terminates after 2 steps, env 1 after 3 steps.
    _install_fake_minari(monkeypatch, _FakeMinariDataset([2, 3]))
    cfg = MinariEnvConfig(dataset_id="fake/dataset-v0", num_envs=2, eval_env=False, device="cpu")
    env = make_minari_env(cfg)
    env.reset()

    actions = torch.zeros(2, 1)
    env.step(actions)  # step 1
    _, _, terminations, _, infos = env.step(actions)  # step 2: only env 0 terminates

    assert torch.equal(terminations, torch.tensor([True, False]))
    assert torch.equal(infos["_final_info"], torch.tensor([True, False]))
    # Only env 0's final_observation slot is meaningful; env 0 terminated at
    # local step 2 with obs_start=0 -> final obs value 2.0.
    assert infos["final_observation"].flatten()[0].item() == 2.0


def test_backend_make_eval_env_uses_num_eval_envs_and_eval_flag(monkeypatch):
    captured = {}

    def _fake_make_minari_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-eval-env"

    monkeypatch.setattr("rl_garden.envs.minari.env.make_minari_env", _fake_make_minari_env)

    req = EnvRequest(
        env_id="fake/dataset-v0",
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
    result = MinariBackend.make_eval_env(req)
    assert result == "sentinel-eval-env"
    cfg = captured["cfg"]
    assert cfg.dataset_id == "fake/dataset-v0"
    assert cfg.num_envs == 2
    assert cfg.eval_env is True


def test_backend_make_train_env_uses_num_envs_and_train_flag(monkeypatch):
    captured = {}

    def _fake_make_minari_env(cfg):
        captured["cfg"] = cfg
        return "sentinel-train-env"

    monkeypatch.setattr("rl_garden.envs.minari.env.make_minari_env", _fake_make_minari_env)

    req = EnvRequest(
        env_id="fake/dataset-v0",
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
    result = MinariBackend.make_train_env(req)
    assert result == "sentinel-train-env"
    cfg = captured["cfg"]
    assert cfg.num_envs == 4
    assert cfg.eval_env is False
