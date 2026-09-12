"""OGBench environment factory.

Every OGBench task variant (locomotion: pointmaze/antmaze/humanoidmaze/
antsoccer; manipulation: cube/scene/puzzle; state or pixel observations) is a
distinct env id registered with ``gymnasium.envs.registration.register`` as
an import side-effect of ``import ogbench`` -- there is no per-family
branching or metadata-driven construction needed here, unlike the
``robomimic`` backend. ``gymnasium.make(env_id)`` alone reproduces the exact
task, so this module is structurally a near-copy of
``rl_garden.envs.mujoco.env`` (same ``TorchVectorEnvAdapter`` /
``RecordEpisodeStatistics`` / sync-vs-async-vectorization shape); see that
module's docstring for the ``final_observation``/``final_info`` contract and
the headless-rendering rationale.

Pixel observations (``visual-*`` env ids) come back from OGBench as a single
flat ``Box(0, 255, (H, W, C), uint8)`` per env; ``_PixelDictWrapper`` below
renames it into the observation contract's Dict shape,
``Dict({"rgb_<camera>": Box})``, keyed by the single camera name
``ObservationConfig.rgb`` asked for (OGBench has no named cameras of its
own). ``cfg.frame_stack > 1`` then applies rl_garden's ``ImageFrameStackWrapper``
to that ``rgb_<camera>`` key like any other Dict-keyed image backend. Set
``vectorization="async"`` for ``visual-*`` env ids so each instance owns its
own MuJoCo renderer/GL context (same reasoning as the ``mujoco`` backend's
own vectorization knob).
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.envs.ogbench.config import OGBenchEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


class _PixelDictWrapper(gym.ObservationWrapper):
    """Renames a ``visual-*`` env id's flat pixel ``Box`` into
    ``Dict({"rgb_<camera>": Box})`` -- the observation contract's Dict shape,
    keyed by the single camera name ``ObservationConfig.rgb`` asked for
    (OGBench has no named cameras of its own; the requested name becomes the
    key)."""

    def __init__(self, env: gym.Env, *, camera: str) -> None:
        super().__init__(env)
        self._camera = camera
        self.observation_space = spaces.Dict({f"rgb_{camera}": env.observation_space})

    def observation(self, observation):
        return {f"rgb_{self._camera}": observation}


class _OGBenchEpisodeMetrics(gym.Wrapper):
    """Attach ``success_at_end`` to terminal episode statistics.

    Every OGBench task family sets a scalar ``info["success"]`` on every step
    (verified directly in ``ogbench``'s ``locomaze``/``manipspace`` env
    source), but ``gymnasium.wrappers.RecordEpisodeStatistics`` only tracks
    return/length, so it never reaches ``run_exact_episode_eval``'s
    ``episode`` dict without this -- mirrors
    ``rl_garden.envs.d4rl_legacy.env._D4RLEpisodeMetrics`` /
    ``rl_garden.envs.robomimic.env._RobomimicEpisodeMetrics``.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and "episode" in info:
            info = dict(info)
            episode = dict(info["episode"])
            episode["success_at_end"] = np.float32(info["success"])
            info["episode"] = episode
        return obs, reward, terminated, truncated, info


class _SeededOGBenchVecEnv(TorchVectorEnvAdapter):
    """Defaults ``reset()`` to the configured ``seed`` when the caller omits
    one, so repeated evaluation rounds replay the same fixed episodes
    (mirrors ``rl_garden.envs.mujoco.env._SeededMujocoVecEnv`` -- duplicated
    rather than shared since it's a module-private helper, not a
    subclassing hook)."""

    def __init__(self, vec_env: gym.vector.VectorEnv, device: str, seed: int) -> None:
        super().__init__(vec_env, device)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=seed if seed is not None else self._seed, options=options)


def _make_env_fn(env_id: str, env_kwargs: dict[str, Any], pixel_camera: str | None):
    def _env_fn():
        # Registers every OGBench env id as an import side-effect; needed
        # fresh inside AsyncVectorEnv spawn workers, which don't inherit
        # whatever the parent process already imported.
        import ogbench  # noqa: F401

        # Default to the GPU-accelerated headless backend before any
        # rendering happens (relevant for visual-* env ids); setdefault so
        # an explicit caller override (e.g. MUJOCO_GL=osmesa) still wins.
        os.environ.setdefault("MUJOCO_GL", "egl")

        from gymnasium.wrappers import RecordEpisodeStatistics

        base_env = gym.make(env_id, **env_kwargs)
        if pixel_camera is not None:
            base_env = _PixelDictWrapper(base_env, camera=pixel_camera)
        else:
            from rl_garden.envs.wrappers import DictStateObservationWrapper

            base_env = DictStateObservationWrapper(base_env)
        env = RecordEpisodeStatistics(base_env)
        return _OGBenchEpisodeMetrics(env)

    return _env_fn


def make_ogbench_env(cfg: OGBenchEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [
        _make_env_fn(cfg.env_id, cfg.env_kwargs, cfg.pixel_camera) for _ in range(cfg.num_envs)
    ]
    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededOGBenchVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.frame_stack > 1:
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        adapter = ImageFrameStackWrapper(
            adapter, frame_stack=cfg.frame_stack, image_keys=(f"rgb_{cfg.pixel_camera}",)
        )

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
