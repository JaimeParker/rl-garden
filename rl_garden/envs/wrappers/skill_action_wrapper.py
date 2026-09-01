"""Batched skill-macro-action wrapper for GPU-vectorized, SAME_STEP-autoreset
envs -- SUPE's online half.

``action`` (the input to ``step()``) has shape ``(num_envs, skill_dim)``: one
skill held fixed across ``horizon`` raw sub-steps while ``decoder`` (OPAL's
frozen decoder, ``OPALVAE.decoder`` -- a ``UnsquashedGaussianActor``, already
shipped) produces a fresh raw action from ``(obs_t, skill)`` at each
sub-step. Same batched-heterogeneous-termination handling as
``ActionChunkWrapper`` (``action_chunk.py`` -- see its docstring for the
accepted post-autoreset tradeoff: a terminated slot keeps receiving decoded
actions for the rest of the chunk, unavoidable under SAME_STEP autoreset,
but the returned macro-transition's reward/final-obs are unaffected), reusing
the same ``_tree_where`` helper. The one structural difference from
``ActionChunkWrapper`` is this wrapper decodes each sub-step's action itself
from the *current* observation, rather than consuming a pre-computed chunk
from the caller.

``deterministic`` is fixed at construction (mirrors upstream SUPE's separate
``sample_skill_actions``/``eval_skill_actions``,
``SUPE/supe/pretraining/opal.py:447-460``): False for the training
env (stochastic decode via ``decoder.action_log_prob``'s rsample+clamp
path), True for eval (``decoder.deterministic_action``'s mean+clamp path).

State-only (Box observations): ``features = concat(obs, skill)`` assumes a
flat observation tensor, matching every other algorithm's state-only scope.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.common.types import Obs
from rl_garden.envs.wrappers._batch_utils import _tree_where
from rl_garden.networks.actor_critic import UnsquashedGaussianActor


class SkillActionWrapper(gym.Wrapper):
    """``action`` has shape ``(num_envs, skill_dim)``.

    Reward is summed across the horizon; ``terminated``/``truncated`` are
    combined with logical-or across sub-steps.
    """

    def __init__(
        self,
        env: gym.Env,
        decoder: UnsquashedGaussianActor,
        *,
        horizon: int,
        skill_dim: int,
        deterministic: bool = False,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if not isinstance(env.single_observation_space, spaces.Box):
            raise TypeError("SkillActionWrapper supports Box observation spaces only.")
        super().__init__(env)
        self.decoder = decoder
        self.horizon = int(horizon)
        self.deterministic = deterministic
        # gym.Wrapper has no __getattr__ fallback for these rl-garden/ManiSkill
        # vector-env conventions -- same reasoning as ActionChunkWrapper.
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        # Preserved for callers (SUPE.__init__ reconstructs its own frozen
        # OPALVAE from an opal_checkpoint path) that need the true action
        # bounds after this wrapper overwrites single_action_space/
        # action_space below with the skill-space contract.
        self.raw_action_space = env.single_action_space
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(skill_dim,), dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, env.num_envs)
        self._last_obs: Obs | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, skill: torch.Tensor):
        num_envs = skill.shape[0]
        device = skill.device
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        reward_sum = torch.zeros(num_envs, device=device)
        chunk_terminated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        chunk_truncated = torch.zeros(num_envs, dtype=torch.bool, device=device)
        final_observation: Obs | None = None
        final_info: dict[str, Any] = {}
        obs: Obs
        infos: dict[str, Any] = {}

        for _ in range(self.horizon):
            features = torch.cat([self._last_obs, skill], dim=-1)
            with torch.no_grad():
                if self.deterministic:
                    raw_action = self.decoder.deterministic_action(features)
                else:
                    raw_action, _ = self.decoder.action_log_prob(features)
            obs, reward, terminated, truncated, infos = self.env.step(raw_action)
            self._last_obs = obs
            done = terminated | truncated
            newly_done = active & done
            if newly_done.any():
                step_final_obs = infos.get("final_observation", obs)
                final_observation = (
                    step_final_obs
                    if final_observation is None
                    else _tree_where(newly_done, step_final_obs, final_observation)
                )
                for key, value in infos.get("final_info", {}).items():
                    final_info[key] = (
                        _tree_where(newly_done, value, final_info[key])
                        if key in final_info
                        else value
                    )
                chunk_terminated = chunk_terminated | (terminated & newly_done)
                chunk_truncated = chunk_truncated | (truncated & newly_done)
            reward_sum = reward_sum + reward * active.to(reward.dtype)
            active = active & ~done

        out_infos = dict(infos)
        for key in ("final_observation", "_final_observation", "final_info", "_final_info"):
            out_infos.pop(key, None)
        done_mask = ~active
        if done_mask.any():
            out_infos["final_observation"] = final_observation
            out_infos["_final_observation"] = done_mask
            if final_info:
                out_infos["final_info"] = final_info
                out_infos["_final_info"] = done_mask
        return obs, reward_sum, chunk_terminated, chunk_truncated, out_infos
