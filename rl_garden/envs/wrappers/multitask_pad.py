"""``MultitaskPadWrapper``: present one task's vector env in a multitask
model's padded shapes (TD-MPC2 ``MultitaskWrapper._pad_obs``/``step``).

- ``obs["state"]`` (and ``infos["final_observation"]["state"]``) is
  zero-padded on the last dim from the task's own dim to ``obs_dim``.
- The action space becomes ``(action_dim,)``; ``step`` keeps only the
  task's own leading action dims.
- ``infos["task"]`` carries the task id, ``(num_envs,)`` long.

The task's true dims must equal the dataset manifest's dims for that task
(``task_obs_dim``/``task_action_dim``); a mismatch means the env's task or
observation layout differs from the one the model was trained on.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

_VectorWrapperBase = getattr(gym.vector, "VectorWrapper", None) or gym.vector.VectorEnvWrapper


class MultitaskPadWrapper(_VectorWrapperBase):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        *,
        obs_dim: int,
        action_dim: int,
        task_idx: int,
        task_obs_dim: int,
        task_action_dim: int,
    ) -> None:
        super().__init__(env)
        obs_space = env.single_observation_space
        if not isinstance(obs_space, spaces.Dict) or "state" not in obs_space.spaces:
            raise ValueError("MultitaskPadWrapper needs a Dict observation space with a 'state' key.")
        env_obs_dim = obs_space["state"].shape[-1]
        env_action_dim = env.single_action_space.shape[-1]
        if (env_obs_dim, env_action_dim) != (task_obs_dim, task_action_dim):
            raise ValueError(
                f"Env dims (obs={env_obs_dim}, action={env_action_dim}) do not match the "
                f"dataset's task dims (obs={task_obs_dim}, action={task_action_dim})."
            )
        if obs_dim < env_obs_dim or action_dim < env_action_dim:
            raise ValueError(
                f"Padded dims (obs={obs_dim}, action={action_dim}) are smaller than the "
                f"env's (obs={env_obs_dim}, action={env_action_dim})."
            )
        self.padded_obs_dim = int(obs_dim)
        self.padded_action_dim = int(action_dim)
        self.task_idx = int(task_idx)
        self.task_action_dim = int(env_action_dim)

        state_space = obs_space["state"]
        self.single_observation_space = spaces.Dict(
            {
                **obs_space.spaces,
                "state": spaces.Box(-np.inf, np.inf, (self.padded_obs_dim,), dtype=state_space.dtype),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (self.padded_action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, env.num_envs)

    def _pad(self, obs):
        if not isinstance(obs, dict) or "state" not in obs:
            return obs
        state = obs["state"]
        if isinstance(state, np.ndarray):
            width = [(0, 0)] * (state.ndim - 1) + [(0, self.padded_obs_dim - state.shape[-1])]
            padded = np.pad(state, width)
        else:
            padded = F.pad(state, (0, self.padded_obs_dim - state.shape[-1]))
        return {**obs, "state": padded}

    def _task_info(self, infos: dict) -> dict:
        infos = dict(infos)
        infos["task"] = torch.full((self.num_envs,), self.task_idx, dtype=torch.long)
        if "final_observation" in infos:
            infos["final_observation"] = self._pad(infos["final_observation"])
        return infos

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._pad(obs), self._task_info(infos)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions[..., : self.task_action_dim])
        return self._pad(obs), rewards, terminations, truncations, self._task_info(infos)
