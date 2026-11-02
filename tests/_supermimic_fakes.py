"""Fakes for the SuperMimic tests: a planar point mass tracking a circle."""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space

from rl_garden.envs.episode_statistics import EpisodeStatistics
from rl_garden.world_models.state_space import ResidualStateCodec

_REF_LENGTH = 64


class PointMassCodec(ResidualStateCodec):
    """State ``[x, y, vx, vy]`` with residual integration."""

    def __init__(self) -> None:
        super().__init__(4, error_floor=torch.full((4,), 1e-3))


class PointMassMDP:
    """Differentiable analytic MDP: track ``ref(t) = [cos(0.1 t), sin(0.1 t)]``."""

    obs_dim = 4
    goal_dim = 4

    def __init__(self, ref_length: int = _REF_LENGTH) -> None:
        angle = 0.1 * torch.arange(ref_length, dtype=torch.float32)
        self.ref_table = torch.stack([angle.cos(), angle.sin()], dim=-1)
        self.obs_noise: Optional[torch.Tensor] = torch.full((4,), 0.01)

    def ref(self, t: torch.Tensor) -> torch.Tensor:
        table = self.ref_table.to(t.device)
        return table[t.clamp(0, table.shape[0] - 1)]

    def observation(self, s, prev_action, ctx, t):
        return s

    def reward(self, s, action, s_next, prev_action, ctx, t_next):
        tracking = (s_next[..., :2] - self.ref(t_next)).square().sum(-1)
        return torch.exp(-tracking) - 0.01 * action.square().sum(-1)

    def termination_prob(self, s, t):
        return torch.sigmoid(((s[..., :2] - self.ref(t)).norm(dim=-1) - 1.0) / 0.05)

    def goal(self, t):
        return torch.cat([self.ref(t + 1), self.ref(t + 2)], dim=-1)

    def critic_features(self, s, ctx, t):
        return s[..., :2] - self.ref(t)

    def valid_start(self, t, horizon: int):
        return t + horizon < self.ref_table.shape[0]


class PointMassTrackingEnv(gym.vector.VectorEnv):
    """Torch vector env: a point mass ``[x, y, vx, vy]`` with per-episode drag
    tracks ``PointMassMDP.ref``. SAME_STEP autoreset with
    ``final_observation``/``final_info``/``_final_info`` on done rows.

    Observation keys: ``state`` (the state plus uniform ``obs_noise``) and,
    with ``extra_keys``, ``state_sim`` (exact state), ``state_ctx`` (drag),
    ``state_goal`` (``mdp.goal(t)``) and ``state_time`` (``t``, omitted when
    ``time_key=False``). Only an env with ``extra_keys`` exposes
    ``analytic_mdp``.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs: int,
        *,
        device: str = "cpu",
        extra_keys: bool = True,
        time_key: bool = True,
        max_episode_steps: int = 50,
        dt: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.extra_keys = extra_keys
        self.time_key = extra_keys and time_key
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        # Long enough that every start t < T_ref - 2 * max_episode_steps exists.
        self._mdp = PointMassMDP(ref_length=max(_REF_LENGTH, 4 * max_episode_steps))
        self.ref_length = self._mdp.ref_table.shape[0]
        if extra_keys:
            self.analytic_mdp = self._mdp

        def box(dim: int) -> spaces.Box:
            return spaces.Box(-float("inf"), float("inf"), (dim,), dtype="float32")

        keys = {"state": box(4)}
        if extra_keys:
            keys.update(state_sim=box(4), state_ctx=box(1), state_goal=box(PointMassMDP.goal_dim))
        if self.time_key:
            keys["state_time"] = box(1)
        self.single_observation_space = spaces.Dict(keys)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype="float32")
        self.action_space = batch_space(self.single_action_space, num_envs)

        self._generator = torch.Generator(device=self.device).manual_seed(seed)
        self._episode_stats = EpisodeStatistics(num_envs, self.device)
        self._p = torch.zeros(num_envs, 2, device=self.device)
        self._v = torch.zeros(num_envs, 2, device=self.device)
        self._drag = torch.ones(num_envs, 1, device=self.device)
        self._t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._episode_len = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._prev_action = torch.zeros(num_envs, 2, device=self.device)
        self._reset_rows(torch.ones(num_envs, dtype=torch.bool, device=self.device))

    def _uniform(self, *shape: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, generator=self._generator, device=self.device)

    def _reset_rows(self, rows: torch.Tensor) -> None:
        n = self.num_envs
        start_t = torch.randint(
            0, self.ref_length - 2 * self.max_episode_steps, (n,), generator=self._generator, device=self.device
        )
        start_p = self._mdp.ref(start_t) + self._uniform(n, 2, low=-0.05, high=0.05)
        drag = self._uniform(n, 1, low=0.5, high=1.5)
        column = rows.unsqueeze(-1)
        self._t = torch.where(rows, start_t, self._t)
        self._p = torch.where(column, start_p, self._p)
        self._v = torch.where(column, torch.zeros_like(self._v), self._v)
        self._drag = torch.where(column, drag, self._drag)
        self._episode_len = torch.where(rows, torch.zeros_like(self._episode_len), self._episode_len)
        self._prev_action = torch.where(column, torch.zeros_like(self._prev_action), self._prev_action)

    def _state(self) -> torch.Tensor:
        return torch.cat([self._p, self._v], dim=-1)

    def _obs(self) -> dict[str, torch.Tensor]:
        s = self._state()
        noise = self._mdp.obs_noise.to(self.device)
        obs = {"state": s + noise * self._uniform(*s.shape, low=-1.0, high=1.0)}
        if self.extra_keys:
            obs.update(
                state_sim=s.clone(),
                state_ctx=self._drag.clone(),
                state_goal=self._mdp.goal(self._t),
            )
        if self.time_key:
            obs["state_time"] = self._t.float().unsqueeze(-1)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        if seed is not None:
            self._generator.manual_seed(seed)
        self._reset_rows(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._episode_stats.reset()
        return self._obs(), {}

    def step(self, actions: torch.Tensor):
        action = actions.to(self.device).float()
        s = self._state()
        self._v = self._v + (action - self._drag * self._v) * self.dt
        self._p = self._p + self._v * self.dt
        self._t = self._t + 1
        self._episode_len = self._episode_len + 1
        s_next = self._state()
        reward = self._mdp.reward(s, action, s_next, self._prev_action, self._drag, self._t)
        self._prev_action = action
        terminated = (self._p - self._mdp.ref(self._t)).norm(dim=-1) > 1.0
        truncated = (self._episode_len >= self.max_episode_steps) | (self._t + 1 >= self.ref_length - 1)
        done = terminated | truncated
        episode = self._episode_stats.step(reward, done)
        obs = self._obs()

        infos: dict[str, Any] = {}
        if bool(done.any()):
            infos["final_observation"] = {key: value.clone() for key, value in obs.items()}
            infos["final_info"] = {"episode": episode}
            infos["_final_info"] = done
            self._reset_rows(done)
            reset_obs = self._obs()
            obs = {key: torch.where(done.unsqueeze(-1), reset_obs[key], value) for key, value in obs.items()}
        return obs, reward, terminated, truncated, infos

    def close(self) -> None:
        pass
