"""``make_mujoco_warp_env`` wrapper composition, on a stand-in task (a torch
``VectorEnv``) so neither mujoco_warp nor a GPU is needed."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import ClassVar

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

NUM_ENVS = 2


class _CameraTask(VectorEnv):
    metadata: ClassVar[dict] = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, nworld: int, device: str, **render_kwargs) -> None:
        self.num_envs = nworld
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (2,), np.float32),
                "rgb_main": spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, nworld)
        self.single_action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = batch_space(self.single_action_space, nworld)

    def _obs(self):
        return {
            "state": torch.zeros(self.num_envs, 2),
            "rgb_main": torch.zeros(self.num_envs, 4, 4, 3, dtype=torch.uint8),
        }

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, actions):
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), torch.ones(self.num_envs), done, done, {}


@pytest.fixture
def warp_env_module(monkeypatch):
    faked = [
        name for name in ("mujoco_warp", "warp") if importlib.util.find_spec(name) is None
    ]
    for name in faked:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    pytest.importorskip("mujoco")
    before = set(sys.modules)
    module = importlib.import_module("rl_garden.envs.mujoco_warp.env")
    importlib.import_module("rl_garden.envs.mujoco_warp.tasks")
    monkeypatch.setitem(module._TASKS, "camera-task", _CameraTask)
    monkeypatch.setattr(module.CustomMujocoWarpEnv, "CAMERA_NAME", "main")
    yield module
    if faked:
        for name in set(sys.modules) - before:
            if name.startswith("rl_garden.envs.mujoco_warp"):
                del sys.modules[name]


def test_frame_stack_with_reward_transform_scales_rewards_and_stacks_frames(warp_env_module):
    from rl_garden.envs.mujoco_warp.config import MujocoWarpEnvConfig

    env = warp_env_module.make_mujoco_warp_env(
        MujocoWarpEnvConfig(
            env_id="camera-task",
            num_envs=NUM_ENVS,
            seed=0,
            device="cpu",
            render_width=4,
            render_height=4,
            frame_stack=3,
            reward_scale=2.0,
            reward_bias=1.0,
        )
    )
    assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    env.reset()
    obs, rewards, *_ = env.step(torch.zeros(NUM_ENVS, 1))
    assert obs["rgb_main"].shape == (NUM_ENVS, 3, 4, 4, 3)
    assert rewards.tolist() == [3.0, 3.0]  # 1.0 * 2.0 + 1.0
    env.close()
