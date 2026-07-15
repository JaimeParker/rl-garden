"""Episode-strict windowed replay buffer for TD-MPC2.

TD-MPC2's world model trains on contiguous ``horizon+1``-length observation
windows (for latent-rollout consistency, reward, and value losses), unlike the
single-transition sampling every other replay buffer in this package performs.
This is structurally closest to ``TransformerReplayBuffer`` (stride-1 fixed
windows, ``episode_ends``/``_ep_id``/``_step_id`` bookkeeping, ``DictArray``
for Box/Dict observations) but with a stricter validity rule: that buffer's
``RecurrentSamplingMixin._valid_window_batch`` *tolerates* an episode boundary
inside the window (masked downstream via ``episode_starts``); TD-MPC2 needs
any boundary-crossing window rejected outright, since its consistency/value
losses assume every step in the window belongs to the same rollout.

Consequence of that stricter rule (and of not carrying a final-obs side table
like ``FinalObsTableMixin``): the observation slot written immediately after
an episode's last transition holds the *next* episode's reset observation
(gymnasium vector-env autoreset), so any window whose action range would need
to reach into that slot gets rejected by the ``_ep_id``/``_step_id``
contiguity check below. A small fraction of steps at the tail of every
episode is therefore never sampled — negligible when episodes end by
truncation (TD-MPC2's non-episodic default), but it means the ``terminated``
field a caller trains a termination classifier on will never contain a
positive example. See ``TDMPC2``'s ``episodic`` handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces

from rl_garden.buffers._episode_slice_sampling import EpisodeSliceSamplingMixin
from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray, _tree_to_device
from rl_garden.common.obs_utils import index_obs
from rl_garden.common.types import Obs


@dataclass
class EpisodeSliceBufferSample:
    obs: Obs                   # (horizon+1, B, *obs_shape)
    action: torch.Tensor        # (horizon, B, act_dim)
    reward: torch.Tensor        # (horizon, B)
    terminated: torch.Tensor    # (horizon, B), bool


class EpisodeSliceBuffer(EpisodeSliceSamplingMixin, BaseReplayBuffer):
    """Fixed-``horizon`` window buffer that never samples across episodes.

    Layout: ``(per_env_buffer_size, num_envs, ...)``, same ring-buffer
    convention as every other buffer in this package.
    """

    def __init__(
        self,
        observation_space: spaces.Box | spaces.Dict,
        action_space: spaces.Box,
        num_envs: int,
        buffer_size: int,
        horizon: int,
        storage_device: torch.device | str = "cuda",
        sample_device: torch.device | str = "cuda",
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.per_env_buffer_size = buffer_size // num_envs
        if self.per_env_buffer_size <= horizon:
            raise ValueError(
                "per_env_buffer_size "
                f"({self.per_env_buffer_size}) must be > horizon ({horizon})."
            )
        self.horizon = horizon

        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.pos = 0
        self.full = False

        shape = (self.per_env_buffer_size, num_envs)
        self._is_dict_obs = isinstance(observation_space, spaces.Dict)
        if self._is_dict_obs:
            self.obs = DictArray(shape, observation_space, device=self.storage_device)
        elif isinstance(observation_space, spaces.Box):
            self.obs = torch.zeros(
                shape + tuple(observation_space.shape),
                dtype=torch.float32,
                device=self.storage_device,
            )
        else:
            raise TypeError(
                "EpisodeSliceBuffer supports Box or Dict observations, got "
                f"{type(observation_space)}."
            )
        self.actions = torch.zeros(
            shape + tuple(action_space.shape), dtype=torch.float32, device=self.storage_device
        )
        self.rewards = torch.zeros(shape, dtype=torch.float32, device=self.storage_device)
        self.dones = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)
        self.episode_ends = torch.zeros(shape, dtype=torch.bool, device=self.storage_device)

        # Episode-contiguity bookkeeping, same fields/semantics reused from
        # nstep_buffer.py / transformer_replay_buffer.py.
        self._ep_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_ep_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)
        self._step_id = torch.full(shape, -1, dtype=torch.long, device=self.storage_device)
        self._current_step_id = torch.zeros(num_envs, dtype=torch.long, device=self.storage_device)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add(
        self,
        obs: Obs,
        next_obs: Obs,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        episode_end: torch.Tensor,
    ) -> None:
        # next_obs is accepted for interface parity with every other buffer's
        # add() call site, but never stored: for a non-boundary step, obs[t+1]
        # already equals next_obs[t] once the following add() runs; for the
        # episode's last step, no window can read that slot anyway (see
        # module docstring), so persisting it would be wasted work.
        del next_obs

        if self.storage_device.type == "cpu":
            obs = _tree_to_device(obs, self.storage_device) if self._is_dict_obs else obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()
            episode_end = episode_end.cpu()

        done_bool = done.to(self.storage_device).bool().reshape(self.num_envs)
        episode_end_bool = episode_end.to(self.storage_device).bool().reshape(self.num_envs)

        if self._is_dict_obs:
            assert isinstance(obs, dict)
            self.obs[self.pos] = {k: v.to(self.storage_device) for k, v in obs.items()}
        else:
            assert isinstance(obs, torch.Tensor)
            self.obs[self.pos] = obs.to(self.storage_device)
        self.actions[self.pos] = action.to(self.storage_device)
        self.rewards[self.pos] = reward.reshape(self.num_envs).to(self.storage_device)
        self.dones[self.pos] = done_bool
        self.episode_ends[self.pos] = episode_end_bool

        self._ep_id[self.pos] = self._current_ep_id
        self._step_id[self.pos] = self._current_step_id

        self._current_ep_id = self._current_ep_id + episode_end_bool.long()
        self._current_step_id = self._current_step_id + 1

        self._advance()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    # _valid_window_batch / _sample_valid_window_starts / _gather_window are
    # inherited from EpisodeSliceSamplingMixin.

    def sample(self, batch_size: int) -> EpisodeSliceBufferSample:
        t0, env_inds = self._sample_valid_window_starts(batch_size)
        idx_grid, env_grid = self._gather_window(t0, env_inds)

        window_obs = index_obs(self.obs, (idx_grid, env_grid))
        action = self.actions[idx_grid[:-1], env_grid[:-1]]
        reward = self.rewards[idx_grid[:-1], env_grid[:-1]]
        terminated = self.dones[idx_grid[:-1], env_grid[:-1]]

        if self._is_dict_obs:
            obs = _tree_to_device(window_obs, self.sample_device)
        else:
            obs = window_obs.to(self.sample_device)

        return EpisodeSliceBufferSample(
            obs=obs,
            action=action.to(self.sample_device),
            reward=reward.to(self.sample_device),
            terminated=terminated.to(self.sample_device),
        )
