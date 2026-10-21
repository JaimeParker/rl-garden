"""Stand-in for a user's own env module, for ``tests/test_gymnasium_backend.py``.

Importing this module registers two gymnasium ids, the way a user's package
would when listed in ``--gymnasium.import_modules``:

- ``RlgTestPixels-v0``: Dict observation already keyed by the rl-garden
  contract (``rgb_front`` uint8 16x16x3 + ``state`` float32 (2,)), reward 1.0
  per step, truncated after 3 steps.
- ``RlgTestPreWrapped-v0``: the same env, whose entry point already applies
  ``RecordEpisodeStatistics`` outside its own 3-step ``TimeLimit`` (so the
  statistics wrapper sees the truncation).
- ``RlgTestOddKeys-v0``: the same env with non-contract keys ``pixels`` and
  ``proprio``, for ``obs_key_map``.

It also defines ``vector_env_factory`` targets: ``make_numpy_vec`` (numpy
SyncVectorEnv), ``make_torch_vec`` (torch-native ``TorchCountingVecEnv``),
``make_next_step_vec`` (wrong autoreset mode) and ``make_box_vec`` (bare Box
observation).
"""
from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

EPISODE_STEPS = 3


class PixelsStateEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, image_key: str = "rgb_front", state_key: str = "state") -> None:
        self._image_key = image_key
        self._state_key = state_key
        self.observation_space = spaces.Dict(
            {
                image_key: spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
                state_key: spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def step(self, action):
        return self._observation(), 1.0, False, False, {}

    def _observation(self):
        return {
            self._image_key: self.np_random.integers(0, 256, size=(16, 16, 3), dtype=np.uint8),
            self._state_key: self.np_random.standard_normal(2).astype(np.float32),
        }


def make_pre_wrapped_env() -> gym.Env:
    return RecordEpisodeStatistics(
        TimeLimit(PixelsStateEnv(), max_episode_steps=EPISODE_STEPS)
    )


def _sync_vec(env_id: str, num_envs: int, autoreset_mode: AutoresetMode) -> SyncVectorEnv:
    return SyncVectorEnv(
        [lambda: RecordEpisodeStatistics(gym.make(env_id)) for _ in range(num_envs)],
        autoreset_mode=autoreset_mode,
    )


def make_numpy_vec(num_envs, seed, device, env_id="RlgTestPixels-v0"):
    return _sync_vec(env_id, num_envs, AutoresetMode.SAME_STEP)


def make_next_step_vec(num_envs, seed, device):
    return _sync_vec("RlgTestPixels-v0", num_envs, AutoresetMode.NEXT_STEP)


def make_box_vec(num_envs, seed, device):
    return _sync_vec("Pendulum-v1", num_envs, AutoresetMode.SAME_STEP)


class TorchCountingVecEnv(VectorEnv):
    """Torch-native vector env: ``state`` is drawn from the reset seed, reward
    1.0 per step, every env truncates after ``EPISODE_STEPS`` steps (SAME_STEP
    autoreset with ``final_observation`` and ``final_info["episode"]``)."""

    metadata: ClassVar[dict] = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, num_envs: int, device: str) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)}
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.reset_seeds: list[int | None] = []
        self._generator = torch.Generator()
        self._elapsed = torch.zeros(num_envs, dtype=torch.long)

    def _draw_state(self, count: int) -> torch.Tensor:
        return torch.randn((count, 2), generator=self._generator).to(self.device)

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)
        self._elapsed.zero_()
        self._state = self._draw_state(self.num_envs)
        return {"state": self._state.clone()}, {}

    def step(self, actions):
        self._elapsed += 1
        reward = torch.ones(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done = self._elapsed >= EPISODE_STEPS
        truncated = done.to(self.device)
        infos = {}
        if bool(done.any()):
            mask = truncated.clone()
            infos = {
                "final_observation": {"state": self._state.clone()},
                "_final_observation": mask,
                "final_info": {
                    "episode": {
                        "r": self._elapsed.float().to(self.device),
                        "l": self._elapsed.to(self.device),
                    }
                },
                "_final_info": mask,
            }
            self._state[truncated] = self._draw_state(int(done.sum()))
            self._elapsed[done] = 0
        return {"state": self._state.clone()}, reward, terminated, truncated, infos


def make_torch_vec(num_envs, seed, device):
    return TorchCountingVecEnv(num_envs, device)


if "RlgTestPixels-v0" not in gym.registry:
    gym.register(
        "RlgTestPixels-v0", entry_point=PixelsStateEnv, max_episode_steps=EPISODE_STEPS
    )
if "RlgTestPreWrapped-v0" not in gym.registry:
    gym.register("RlgTestPreWrapped-v0", entry_point=make_pre_wrapped_env)
if "RlgTestOddKeys-v0" not in gym.registry:
    gym.register(
        "RlgTestOddKeys-v0",
        entry_point=PixelsStateEnv,
        max_episode_steps=EPISODE_STEPS,
        kwargs={"image_key": "pixels", "state_key": "proprio"},
    )
