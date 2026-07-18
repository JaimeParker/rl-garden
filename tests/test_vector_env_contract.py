"""Contract tests for TorchVectorEnvAdapter, plus a conformance check that
ManiSkill's and RoboTwin's own env objects already satisfy the same
algorithm-facing attribute/key surface without needing this adapter."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.envs.vector_env import TorchVectorEnvAdapter


class _TinyEnv(gym.Env):
    """Deterministic single-obs-dim env that terminates after ``terminate_at`` steps."""

    def __init__(self, terminate_at: int, start: float = 0.0):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.terminate_at = terminate_at
        self.start = start
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.array([self.start], dtype=np.float64), {}

    def step(self, action):
        self._t += 1
        obs = np.array([self.start + self._t], dtype=np.float64)
        terminated = self._t >= self.terminate_at
        info = {"episode": {"l": self._t}} if terminated else {}
        return obs, 1.0, terminated, False, info


class _DictObsEnv(gym.Env):
    """Dict-observation-space env, to exercise the per-key stacking path."""

    def __init__(self, terminate_at: int, start: float = 0.0):
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                "aux": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.terminate_at = terminate_at
        self.start = start
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = {
            "state": np.array([self.start], dtype=np.float64),
            "aux": np.array([self.start * 10], dtype=np.float64),
        }
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = {
            "state": np.array([self.start + self._t], dtype=np.float64),
            "aux": np.array([(self.start + self._t) * 10], dtype=np.float64),
        }
        terminated = self._t >= self.terminate_at
        return obs, 1.0, terminated, False, {}


def _make_adapter(terminate_ats, device="cpu"):
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    def make(i):
        return lambda: _TinyEnv(terminate_at=terminate_ats[i], start=i * 100.0)

    vec = SyncVectorEnv(
        [make(i) for i in range(len(terminate_ats))], autoreset_mode=AutoresetMode.SAME_STEP
    )
    return TorchVectorEnvAdapter(vec, device=device)


def test_adapter_is_a_vector_env():
    env = _make_adapter([5, 5])
    assert isinstance(env, gym.vector.VectorEnv)
    assert env.num_envs == 2


def test_reset_and_step_return_torch_tensors_on_configured_device():
    env = _make_adapter([5, 5])
    obs, _ = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.dtype == torch.float32  # translated from the env's float64
    assert obs.shape == (2, 1)
    assert obs.device.type == "cpu"

    actions = torch.zeros(2, 1)
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    assert isinstance(next_obs, torch.Tensor)
    assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    assert isinstance(terminated, torch.Tensor) and terminated.dtype == torch.bool
    assert isinstance(truncated, torch.Tensor) and truncated.dtype == torch.bool


def test_partial_termination_final_observation_matches_maniskill_convention():
    env = _make_adapter([2, 5])  # env0 terminates at step 2, env1 keeps going
    env.reset()
    actions = torch.zeros(2, 1)
    env.step(actions)  # step 1
    obs, rewards, terminated, truncated, infos = env.step(actions)  # step 2

    assert torch.equal(terminated, torch.tensor([True, False]))
    # SAME_STEP autoreset: env0's returned obs is already the reset observation.
    assert obs.flatten()[0].item() == 0.0
    # env1 (not terminated) obs is just its normal current observation.
    assert obs.flatten()[1].item() == 102.0

    # final_observation: env0 gets its true pre-reset terminal obs (2.0);
    # env1 (not masked) gets its actual current obs (102.0), matching
    # ManiSkillVectorEnv.step()'s "just o_{t+1}" convention -- not zeros.
    final_obs = infos["final_observation"]
    assert torch.equal(final_obs.flatten(), torch.tensor([2.0, 102.0]))
    assert torch.equal(infos["_final_observation"], torch.tensor([True, False]))

    # final_info is preserved (the CPU MuJoCo adapter this replaces used to drop it).
    assert torch.equal(infos["_final_info"], torch.tensor([True, False]))
    assert infos["final_info"]["episode"]["l"][0].item() == 2


def test_dict_observation_space_is_translated_and_stacked():
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    vec = SyncVectorEnv(
        [lambda: _DictObsEnv(terminate_at=2), lambda: _DictObsEnv(terminate_at=2)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = TorchVectorEnvAdapter(vec, device="cpu")

    assert env.single_observation_space["state"].dtype == np.float32
    assert env.single_observation_space["aux"].dtype == np.float32

    env.reset()
    actions = torch.zeros(2, 1)
    env.step(actions)  # step 1
    obs, rewards, terminated, truncated, infos = env.step(actions)  # step 2: both terminate

    assert torch.equal(terminated, torch.tensor([True, True]))
    final_obs = infos["final_observation"]
    assert isinstance(final_obs, dict)
    assert torch.equal(final_obs["state"].flatten(), torch.tensor([2.0, 2.0]))
    assert torch.equal(final_obs["aux"].flatten(), torch.tensor([20.0, 20.0]))
    assert final_obs["state"].dtype == torch.float32


def test_maniskill_vector_env_already_conforms():
    pytest.importorskip("mani_skill")
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    assert issubclass(ManiSkillVectorEnv, gym.vector.VectorEnv)


def test_robotwin_env_already_conforms():
    from rl_garden.envs.robotwin.env import RoboTwinEnv

    assert issubclass(RoboTwinEnv, gym.Env)
    assert hasattr(RoboTwinEnv, "step") and hasattr(RoboTwinEnv, "reset")
