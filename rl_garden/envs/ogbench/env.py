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

Pixel observations (``visual-*`` env ids) are a single flat
``Box(0, 255, (H, W, C), uint8)`` per env, not a Dict -- rl_garden's
``ImageFrameStackWrapper`` only supports Dict-keyed ``rgb``/``depth``
observations, so frame stacking is out of scope here; pixel envs pass
through as flat Box observations. Set ``vectorization="async"`` for
``visual-*`` env ids so each instance owns its own MuJoCo renderer/GL
context (same reasoning as the ``mujoco`` backend's own vectorization knob).
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym

from rl_garden.envs.ogbench.config import OGBenchEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


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


def _make_env_fn(env_id: str, env_kwargs: dict[str, Any]):
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

        return RecordEpisodeStatistics(gym.make(env_id, **env_kwargs))

    return _env_fn


def make_ogbench_env(cfg: OGBenchEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [_make_env_fn(cfg.env_id, cfg.env_kwargs) for _ in range(cfg.num_envs)]
    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededOGBenchVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
