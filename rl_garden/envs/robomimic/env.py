"""Adapter for robomimic (robosuite/MuJoCo manipulation) environments.

Not shared with ``rl_garden/envs/d4rl_legacy/env.py``'s ``_GymV21Adapter``:
aside from the 3-line 4->5-tuple translation in ``step()``, everything else
differs -- observation space here must be built by flattening a Dict
(d4rl_legacy's is a native Box/Dict already), action space comes from
``action_dimension`` (an int, already ``[-1, 1]``, no rescale), and
reset/render signatures don't match. See the implementation plan's "Design
decisions" section for the full comparison.
"""
from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.buffers.robomimic_dataset import (
    ROBOMIMIC_LOW_DIM_OBS_KEYS,
    flatten_robomimic_obs,
)
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


def _load_env_meta(cfg) -> dict:
    import robomimic.utils.file_utils as FileUtils

    if cfg.dataset_path is not None:
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset_path)
        dataset_env_name = env_meta.get("env_name")
        if dataset_env_name is not None and dataset_env_name != cfg.env_id:
            raise ValueError(
                f"--env_id={cfg.env_id!r} does not match the dataset's own "
                f"env_name={dataset_env_name!r} (from {cfg.dataset_path!r}). "
                "robomimic's controller/robot config comes entirely from the "
                "dataset's env_args -- pass --env_id matching the dataset, or "
                "omit --robomimic.dataset_path to build the env from "
                "--env_id plus --robomimic.env_kwargs_json defaults instead."
            )
        return env_meta
    return {"env_name": cfg.env_id, "type": 1, "env_kwargs": json.loads(cfg.env_kwargs_json)}


def _make_robosuite_env(cfg) -> Any:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    # robomimic's EnvRobosuite.get_observation() looks up every raw obs key
    # in ObsUtils.OBS_KEYS_TO_MODALITIES (a process-global dict), which is
    # None until this is called -- must run before any env construction, in
    # every process (AsyncVectorEnv subprocesses included, hence calling it
    # here rather than once at import time). Unlisted keys (e.g. robosuite's
    # force-activated extras not in ROBOMIMIC_LOW_DIM_OBS_KEYS) auto-register
    # as "low_dim" on first lookup -- this only needs to name the keys we
    # actually care about.
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {"low_dim": list(ROBOMIMIC_LOW_DIM_OBS_KEYS)}
    )
    env_meta = _load_env_meta(cfg)
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False
    )


class _RobomimicGymV21Adapter(gym.Env):
    """Wrap a robomimic ``EnvBase`` (old 4-tuple gym API, Dict obs) into
    gymnasium's 5-tuple contract with a flat Box observation."""

    def __init__(self, env: Any) -> None:
        super().__init__()
        self.legacy_env = env
        flat0 = flatten_robomimic_obs(env.reset())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat0.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_dimension,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        # robomimic's EnvBase exposes no confirmed seed() equivalent (open
        # item -- see implementation plan); seed is accepted but not forwarded.
        obs = self.legacy_env.reset()
        return flatten_robomimic_obs(obs).astype(np.float32), {}

    def step(self, action):
        obs, reward, done, info = self.legacy_env.step(action)
        # EnvRobosuite.is_done() always returns False (confirmed from
        # robomimic source: robosuite envs always rollout to a fixed
        # horizon) -- terminated is structurally always False here;
        # truncation comes from gym.wrappers.TimeLimit in _make_env_fn.
        return flatten_robomimic_obs(obs).astype(np.float32), reward, bool(done), False, info

    def render(self):
        return None

    def close(self) -> None:
        # robomimic's EnvRobosuite wrapper does not itself expose close() --
        # only the underlying robosuite env (EnvRobosuite.env) does
        # (confirmed by reading robomimic/robosuite source: EnvRobosuite has
        # no close() method at all, unlike D4RL's legacy envs).
        self.legacy_env.env.close()


class _RobomimicSuccessTermination(gym.Wrapper):
    """Opt-in: treat robomimic's task-success signal as true termination.

    ``info["is_success"]`` is a dict (e.g. ``{"task": bool}``), never a bare
    bool -- index ``["task"]`` explicitly, never truthiness-test the dict
    itself (a non-empty dict is always truthy).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = bool(info["is_success"]["task"])
        return obs, reward, terminated or success, truncated, info


class _RobomimicEpisodeMetrics(gym.Wrapper):
    """Attach ``success_at_end`` to terminal episode statistics."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and "episode" in info:
            info = dict(info)
            episode = dict(info["episode"])
            episode["success_at_end"] = np.float32(bool(info["is_success"]["task"]))
            info["episode"] = episode
        return obs, reward, terminated, truncated, info


def _make_env_fn(cfg):
    def _make_env():
        env: gym.Env = _RobomimicGymV21Adapter(_make_robosuite_env(cfg))
        if cfg.terminate_on_success:
            env = _RobomimicSuccessTermination(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.horizon)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return _RobomimicEpisodeMetrics(env)

    return _make_env


def make_robomimic_env(cfg):
    env_fns = [_make_env_fn(cfg) for _ in range(cfg.num_envs)]
    # robosuite/MuJoCo is CPU-only and process-isolated (same story as
    # d4rl_legacy): SyncVectorEnv never parallelizes beyond one process,
    # AsyncVectorEnv gives real speedup above num_envs=1.
    vector_cls = gym.vector.AsyncVectorEnv if cfg.num_envs > 1 else gym.vector.SyncVectorEnv
    vec_env = vector_cls(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = TorchVectorEnvAdapter(vec_env, device=cfg.device)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(
            adapter, scale=cfg.reward_scale, bias=cfg.reward_bias
        )
    return adapter
