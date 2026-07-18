"""MuJoCo environment factory.

Wraps either Gymnasium's registered MuJoCo benchmark tasks (``HalfCheetah-v4``,
``Ant-v4``, ``Hopper-v4``, ...) or rl-garden's own custom tasks (see
``rl_garden.envs.mujoco.custom_mujoco_env.CustomMujocoEnv``) with a Gymnasium
vector env and rl-garden's shared ``TorchVectorEnvAdapter``
(``rl_garden.envs.vector_env`` -- see its module docstring for the
``final_observation``/``final_info`` contract) so the result exposes the
same shape the rest of ``rl_garden`` targets: ``num_envs`` /
``single_observation_space`` / ``single_action_space``, and ``step``/
``reset`` returning torch tensors.

Vectorization: ``cfg.vectorization == "sync"`` (default) uses
``gymnasium.vector.SyncVectorEnv`` (single process, matches the original CPU
v1 path for state-only benchmark tasks). ``"async"`` uses
``gymnasium.vector.AsyncVectorEnv`` (one OS process per env) — required for
custom tasks with camera observations, since a single ``MujocoRenderer``'s
OpenGL context isn't verified safe to share across multiple env instances in
one process; running each env in its own process sidesteps that question by
construction rather than assuming it's fine. The ``env_fns`` thunk
(``_make_env_fn``) imports ``rl_garden.envs.mujoco.tasks`` (triggering
``gym.register()`` for rl-garden's own custom tasks) *inside* the thunk, not
just at module scope — under ``AsyncVectorEnv`` with a spawn-based
multiprocessing start method, worker processes don't inherit whatever the
parent process already imported, so registration has to happen freshly in
whichever process actually calls ``gym.make()``.

Headless rendering: Gymnasium's ``MujocoRenderer`` auto-detects an OpenGL
backend (GLFW / EGL / OSMesa, in that order) by trying each in turn, but on a
display-less machine GLFW can silently "succeed" at context creation and then
fail much later, inside ``mujoco.MjrContext(...)``, with a confusing
``gladLoadGL error`` — instead of cleanly falling through to EGL. The
``_env_fn`` thunk sets ``MUJOCO_GL=egl`` by default (only if the caller
hasn't already set it) before any rendering happens, since EGL is the
GPU-accelerated headless backend and (unlike GLFW) doesn't need a display
server. Set ``MUJOCO_GL=osmesa`` yourself beforehand for CPU-only rendering
on a machine without a working EGL/GPU setup.
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym

from rl_garden.envs.mujoco.config import MujocoEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


class _SeededMujocoVecEnv(TorchVectorEnvAdapter):
    """Defaults ``reset()`` to the configured ``seed`` when the caller omits
    one -- e.g. ``base_algorithm.py``'s ``self.eval_env.reset()`` (no args),
    called once per evaluation round throughout training. Without this,
    each evaluation round would continue the RNG stream from wherever it
    left off instead of always replaying the same fixed episodes."""

    def __init__(self, vec_env: gym.vector.VectorEnv, device: str, seed: int) -> None:
        super().__init__(vec_env, device)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=seed if seed is not None else self._seed, options=options)


def _make_env_fn(env_id: str, env_kwargs: dict[str, Any]):
    def _env_fn():
        # Self-contained: triggers gym.register() for rl-garden's own custom
        # tasks inside whichever process actually calls this (needed for
        # AsyncVectorEnv workers under a spawn start method — see module
        # docstring). Harmless no-op if already imported / env_id isn't one
        # of rl-garden's custom tasks.
        import rl_garden.envs.mujoco.tasks  # noqa: F401

        # Default to the GPU-accelerated headless backend before any
        # rendering happens (see module docstring); setdefault so an
        # explicit caller override (e.g. MUJOCO_GL=osmesa) still wins.
        os.environ.setdefault("MUJOCO_GL", "egl")

        # off_policy.py/base_algorithm.py's rollout/eval logging reads
        # infos["final_info"]["episode"] unconditionally whenever
        # "final_info" is present (matching gymnasium's RecordEpisodeStatistics
        # convention, same as rl_garden.envs.minari.env's identical wrap).
        # Required so that key exists once TorchVectorEnvAdapter starts
        # actually preserving final_info end to end.
        from gymnasium.wrappers import RecordEpisodeStatistics

        return RecordEpisodeStatistics(gym.make(env_id, **env_kwargs))

    return _env_fn


def make_mujoco_env(cfg: MujocoEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [_make_env_fn(cfg.env_id, cfg.env_kwargs) for _ in range(cfg.num_envs)]
    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededMujocoVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
