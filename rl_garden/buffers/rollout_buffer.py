"""Torch-native rollout buffers for on-policy algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from gymnasium import spaces

from rl_garden.buffers.dict_buffer import DictArray
from rl_garden.common.types import Obs


@dataclass
class RolloutBufferSample:
    obs: Obs
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _flatten_obs(obs):
    if isinstance(obs, DictArray):
        return {key: _flatten_obs(value) for key, value in obs.data.items()}
    if isinstance(obs, dict):
        return {key: _flatten_obs(value) for key, value in obs.items()}
    return obs.reshape((-1,) + obs.shape[2:])


def _index_obs(obs, indices: torch.Tensor):
    if isinstance(obs, dict):
        return {key: _index_obs(value, indices) for key, value in obs.items()}
    return obs[indices]


class RolloutBuffer:
    """Fixed-size on-policy buffer with ``(num_steps, num_envs, ...)`` layout."""

    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        num_steps: int,
        num_envs: int,
        device: torch.device | str = "cuda",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if not isinstance(action_space, spaces.Box):
            raise TypeError("RolloutBuffer only supports Box action spaces.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pos = 0
        self.full = False
        self.generator_ready = False

        shape = (num_steps, num_envs)
        if isinstance(observation_space, spaces.Dict):
            self.obs = DictArray(shape, observation_space, device=self.device)
        elif isinstance(observation_space, spaces.Box):
            self.obs = torch.zeros(
                shape + tuple(observation_space.shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise TypeError(
                "RolloutBuffer supports Box or Dict observations, got "
                f"{type(observation_space)}."
            )
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), dtype=torch.float32, device=self.device
        )
        self.log_probs = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.final_values = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(shape, dtype=torch.float32, device=self.device)

    @property
    def buffer_size(self) -> int:
        return self.num_steps * self.num_envs

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.final_values.zero_()

    def add(
        self,
        obs: Obs,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        final_values: torch.Tensor | None = None,
    ) -> None:
        if self.pos >= self.num_steps:
            raise RuntimeError(
                "RolloutBuffer is full; call reset() before adding more."
            )

        if isinstance(self.obs, DictArray):
            assert isinstance(obs, dict)
            self.obs[self.pos] = {
                key: value.to(self.device) for key, value in obs.items()
            }
        else:
            assert isinstance(obs, torch.Tensor)
            self.obs[self.pos] = obs.to(self.device)
        self.actions[self.pos] = actions.to(self.device)
        self.rewards[self.pos] = rewards.reshape(self.num_envs).to(self.device)
        self.dones[self.pos] = dones.reshape(self.num_envs).float().to(self.device)
        self.values[self.pos] = values.reshape(self.num_envs).to(self.device)
        self.log_probs[self.pos] = log_probs.reshape(self.num_envs).to(self.device)
        if final_values is not None:
            self.final_values[self.pos] = final_values.reshape(self.num_envs).to(
                self.device
            )

        self.pos += 1
        self.full = self.pos == self.num_steps
        self.generator_ready = False

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        *,
        finite_horizon_gae: bool = False,
    ) -> None:
        last_values = last_values.reshape(self.num_envs).to(self.device)
        last_dones = last_dones.reshape(self.num_envs).float().to(self.device)
        advantages = torch.zeros_like(self.rewards)

        if finite_horizon_gae:
            lam_coef_sum = torch.zeros(self.num_envs, device=self.device)
            reward_term_sum = torch.zeros(self.num_envs, device=self.device)
            value_term_sum = torch.zeros(self.num_envs, device=self.device)
            for step in reversed(range(self.num_steps)):
                if step == self.num_steps - 1:
                    next_not_done = 1.0 - last_dones
                    next_values = last_values
                else:
                    next_not_done = 1.0 - self.dones[step + 1]
                    next_values = self.values[step + 1]
                real_next_values = next_not_done * next_values + self.final_values[step]
                lam_coef_sum = lam_coef_sum * next_not_done
                reward_term_sum = reward_term_sum * next_not_done
                value_term_sum = value_term_sum * next_not_done
                lam_coef_sum = 1.0 + self.gae_lambda * lam_coef_sum
                reward_term_sum = (
                    self.gae_lambda * self.gamma * reward_term_sum
                    + lam_coef_sum * self.rewards[step]
                )
                value_term_sum = (
                    self.gae_lambda * self.gamma * value_term_sum
                    + self.gamma * real_next_values
                )
                advantages[step] = (
                    reward_term_sum + value_term_sum
                ) / lam_coef_sum - self.values[step]
        else:
            last_gae_lam = torch.zeros(self.num_envs, device=self.device)
            for step in reversed(range(self.num_steps)):
                if step == self.num_steps - 1:
                    next_not_done = 1.0 - last_dones
                    next_values = last_values
                else:
                    next_not_done = 1.0 - self.dones[step + 1]
                    next_values = self.values[step + 1]
                real_next_values = next_not_done * next_values + self.final_values[step]
                delta = (
                    self.rewards[step]
                    + self.gamma * real_next_values
                    - self.values[step]
                )
                last_gae_lam = (
                    delta + self.gamma * self.gae_lambda * next_not_done * last_gae_lam
                )
                advantages[step] = last_gae_lam

        self.advantages = advantages
        self.returns = self.advantages + self.values
        self.generator_ready = False

    def get(self, batch_size: int | None = None) -> Iterator[RolloutBufferSample]:
        if not self.full:
            raise RuntimeError("RolloutBuffer must be full before sampling.")
        if batch_size is None:
            batch_size = self.buffer_size
        flat_obs = _flatten_obs(self.obs)
        flat_actions = self.actions.reshape((-1,) + self.actions.shape[2:])
        flat_values = self.values.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)

        indices = torch.randperm(self.buffer_size, device=self.device)
        for start in range(0, self.buffer_size, batch_size):
            mb_inds = indices[start : start + batch_size]
            yield RolloutBufferSample(
                obs=_index_obs(flat_obs, mb_inds),
                actions=flat_actions[mb_inds],
                old_values=flat_values[mb_inds],
                old_log_prob=flat_log_probs[mb_inds],
                advantages=flat_advantages[mb_inds],
                returns=flat_returns[mb_inds],
            )


class DictRolloutBuffer(RolloutBuffer):
    """Alias class for SB3-style naming when observations are Dict spaces."""
