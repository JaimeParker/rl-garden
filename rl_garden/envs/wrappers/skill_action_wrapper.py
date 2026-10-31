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
the same ``ChunkAccumulator`` (``final_observation`` is reported only if the
inner env reports it; second episode ends within one macro-step are counted
in ``dropped_episode_ends``). The one structural difference from
``ActionChunkWrapper`` is this wrapper decodes each sub-step's action itself
from the *current* observation, rather than consuming a pre-computed chunk
from the caller.

``deterministic`` is fixed at construction: False selects a stochastic
decode (``decoder.action_log_prob``'s rsample+clamp path), True a
deterministic one (``decoder.deterministic_action``'s mean+clamp path).
Upstream SUPE's own online rollout (``MetaPolicyActionWrapper.step()``,
``SUPE/supe/wrappers/meta_env_wrapper.py:104-118``) always calls the
stochastic ``sample_skill_actions`` -- for *both* its train and eval
instances (``train_finetuning_supe.py:185-203``); its own ``eval`` flag
only toggles reward-discount handling, never action determinism. The
deterministic path (``eval_skill_actions``, ``opal.py:454-460``) exists
upstream but is used only by OPAL's own standalone pretraining eval loop,
never by SUPE's online rollout -- so ``rl_garden/training/online/supe.py``
constructs both the train and eval wrapper with ``deterministic=False``,
matching upstream exactly. This parameter stays available as a general
capability of the wrapper, not because SUPE's own pipeline uses ``True``.

State-only: ``features = concat(obs["state"], skill)`` assumes a
``Dict({"state": Box})`` observation (rl-garden's observation contract for
every state-only backend), matching every other algorithm's state-only
scope.

Deliberate simplification vs. upstream: ``MetaPolicyActionWrapper`` isn't
actually a clean one-fresh-rollout-per-``step()`` wrapper -- it keeps a
persistent sliding observation/action window *across* ``step()`` calls
(``meta_env_wrapper.py:177-182``), so a mid-horizon termination can return a
transition whose actions mix the previous and current skill. That
"interpolated" path is gated behind ``FLAGS.interpolate``, which is off by
default with the code's own comment "Not used for any experiments in the
paper" (``train_finetuning_supe.py:43,330``) -- so this wrapper's simpler,
stateless-per-call semantics matches upstream's actual paper-reported
behavior in every case that matters, and a persistent single-instance
sliding window wouldn't vectorize across a batched env anyway.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.common.types import Obs
from rl_garden.envs.wrappers._batch_utils import ChunkAccumulator
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
        if (
            not isinstance(env.single_observation_space, spaces.Dict)
            or set(env.single_observation_space.spaces) != {"state"}
        ):
            raise TypeError(
                "SkillActionWrapper supports state-only Dict({'state': Box}) "
                f"observation spaces only, got {env.single_observation_space!r}."
            )
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
        self._chunk = ChunkAccumulator()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    @property
    def dropped_episode_ends(self) -> torch.Tensor | None:
        return self._chunk.dropped_episode_ends

    def step(self, skill: torch.Tensor):
        self._chunk.begin()
        for _ in range(self.horizon):
            features = torch.cat([self._last_obs["state"], skill], dim=-1)
            with torch.no_grad():
                if self.deterministic:
                    raw_action = self.decoder.deterministic_action(features)
                else:
                    raw_action, _ = self.decoder.action_log_prob(features)
            obs, reward, terminated, truncated, infos = self.env.step(raw_action)
            self._last_obs = obs
            self._chunk.add(reward, terminated, truncated, infos)
        return obs, *self._chunk.finish(obs, infos)
