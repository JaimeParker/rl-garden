"""Torch-native toy vector env reporting ``aux_target`` and ``decision_components``.

Used through rl-garden's generic ``gymnasium`` backend:
``--env-backend gymnasium --gymnasium.vector-env-factory
rlg_downstream_example.toy_env:make_aux_decision_toy_env
--gymnasium.vector-env-returns-torch``.

Each episode draws a goal ``g`` in ``[-0.8, 0.8]^3`` that only the first
observation shows, so the policy has to remember it. Action component ``j``
takes effect on a fixed schedule (components 0-1 on even steps, component 2
every third step, all three at the episode start); the reward is
``-sum_j effective_j * (a_j - g_j)^2 / 3``. ``infos["aux_target"]`` is
``float(g_0 > 0)`` of the episode the acting observation belongs to.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

ACTION_DIM = 3
STATE_DIM = 5


class AuxDecisionToyVecEnv(gym.vector.VectorEnv):
    def __init__(self, num_envs: int, seed: int, device: str, episode_length: int = 12) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-1.0, 1.0, (STATE_DIM,), np.float32)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        # Staggered episode lengths, so the envs do not all reset together.
        self.lengths = episode_length + torch.arange(num_envs, device=self.device) % 4
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    def _sample_goals(self) -> torch.Tensor:
        u = torch.rand((self.num_envs, ACTION_DIM), generator=self._generator, device=self.device)
        return 1.6 * u - 0.8

    def _state(self) -> torch.Tensor:
        shown = (self.steps == 0).float()[:, None]
        progress = (self.steps.float() / self.lengths.float())[:, None]
        return torch.cat([self.goals * shown, shown, progress], dim=-1)

    def _effective_components(self) -> torch.Tensor:
        even = (self.steps % 2 == 0)[:, None].expand(-1, 2)
        third = (self.steps % 3 == 0)[:, None]
        return torch.cat([even, third], dim=-1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._generator.manual_seed(seed)
        self.goals = self._sample_goals()
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        return {"state": self._state()}, {}

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        components = self._effective_components()
        aux_target = (self.goals[:, :1] > 0).float()
        squared_error = (actions - self.goals) ** 2
        rewards = -torch.where(components, squared_error, 0.0).sum(-1) / ACTION_DIM

        self.steps = self.steps + 1
        self.returns = self.returns + rewards
        truncated = self.steps >= self.lengths
        terminated = torch.zeros_like(truncated)
        final_state = self._state()
        episode = {"r": self.returns.clone(), "l": self.steps.float()}

        self.goals = torch.where(truncated[:, None], self._sample_goals(), self.goals)
        self.steps = torch.where(truncated, 0, self.steps)
        self.returns = torch.where(truncated, 0.0, self.returns)
        state = self._state()
        infos = {
            "final_observation": {"state": torch.where(truncated[:, None], final_state, state)},
            "final_info": {"episode": episode},
            "_final_info": truncated,
            "aux_target": aux_target,
            "decision_components": components,
        }
        return {"state": state}, rewards, terminated, truncated, infos


def make_aux_decision_toy_env(num_envs: int, seed: int, device: str, **env_kwargs):
    """``--gymnasium.vector-env-factory`` entry: ``fn(num_envs, seed, device, **env_kwargs)``."""
    return AuxDecisionToyVecEnv(num_envs, seed, device, **env_kwargs)
