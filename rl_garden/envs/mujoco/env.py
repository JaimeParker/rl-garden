"""MuJoCo environment factory.

Wraps either Gymnasium's registered MuJoCo benchmark tasks (``HalfCheetah-v4``,
``Ant-v4``, ``Hopper-v4``, ...) or rl-garden's own custom tasks (see
``rl_garden.envs.mujoco.custom_mujoco_env.CustomMujocoEnv``) with a Gymnasium
vector env and a thin adapter so the result exposes the same shape the rest
of ``rl_garden`` targets: ``num_envs`` / ``single_observation_space`` /
``single_action_space``, and ``step``/``reset`` returning torch tensors.

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

Termination contract: rl_garden's rollout/buffer code
(``rl_garden.algorithms.off_policy``, ``on_policy``, ``sac_core``) expects
SAME_STEP autoreset semantics, matching ``ManiSkillVectorEnv``: the ``obs``
returned on the step where an env terminates/truncates is already the next
episode's reset observation, and the true terminal observation is carried
separately in ``infos["final_observation"]`` (a dense array/tensor, or dict
of such for Dict observation spaces, indexable by the same boolean mask as
``terminated``/``truncated``).

Both ``SyncVectorEnv`` and ``AsyncVectorEnv`` support
``autoreset_mode=SAME_STEP`` and report the terminal observation under
``infos["final_obs"]`` as a ``dtype=object`` NumPy array that is only
populated (non-``None``) at terminated/truncated indices — not a dense array,
and not the key name rl_garden's algorithms look for. ``_rebuild_final_obs``
below bridges that gap for both vectorization modes identically (verified
empirically for both, not assumed identical just because the external
contract matches — see the MuJoCo custom-scaffold design plan).
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_garden.envs.mujoco.config import MujocoEnvConfig


def _torch_dtype_for(np_dtype: np.dtype) -> torch.dtype:
    if np.issubdtype(np_dtype, np.floating):
        return torch.float32
    if np_dtype == np.uint8:
        return torch.uint8
    if np_dtype == np.bool_:
        return torch.bool
    return torch.as_tensor(np.empty(0, dtype=np_dtype)).dtype


def _to_torch(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=_torch_dtype_for(array.dtype), device=device)


def _convert_obs(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, dict):
        return {key: _to_torch(value, device) for key, value in obs.items()}
    return _to_torch(obs, device)


def _translate_box(space: gym.spaces.Box) -> gym.spaces.Box:
    if not np.issubdtype(space.dtype, np.floating):
        # uint8 image spaces etc. pass through unchanged (raw pixel values).
        return space
    # Gymnasium MuJoCo tasks declare float64 Box spaces; rl_garden observations
    # are float32 (matches the cast done in _to_torch).
    return gym.spaces.Box(
        low=space.low.astype(np.float32),
        high=space.high.astype(np.float32),
        shape=space.shape,
        dtype=np.float32,
    )


def _translate_space(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({key: _translate_space(sub) for key, sub in space.spaces.items()})
    if isinstance(space, gym.spaces.Box):
        return _translate_box(space)
    return space


class _MujocoVecEnvAdapter(gym.Env):
    """Adapts a Gymnasium vector env of MuJoCo tasks to the rl-garden env
    shape (see module docstring for the termination/final-obs contract)."""

    metadata = {"render_modes": []}

    def __init__(self, vec_env: gym.vector.VectorEnv, device: str, seed: int) -> None:
        self._vec_env = vec_env
        self.device = torch.device(device)
        self.num_envs = vec_env.num_envs
        self.single_observation_space = _translate_space(vec_env.single_observation_space)
        self.single_action_space = vec_env.single_action_space
        # off_policy.py's random-exploration phase reads the batched
        # action_space.shape directly (not single_action_space).
        self.action_space = vec_env.action_space
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._vec_env.reset(
            seed=seed if seed is not None else self._seed, options=options
        )
        return _convert_obs(obs, self.device), info

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, reward, terminated, truncated, infos = self._vec_env.step(actions_np)
        infos = self._rebuild_final_obs(infos, obs)
        return (
            _convert_obs(obs, self.device),
            _to_torch(reward, self.device),
            _to_torch(terminated, self.device),
            _to_torch(truncated, self.device),
            infos,
        )

    def _rebuild_final_obs(self, infos: dict[str, Any], obs: Any) -> dict[str, Any]:
        final_obs_raw = infos.pop("final_obs", None)
        infos.pop("_final_obs", None)
        infos.pop("final_info", None)
        infos.pop("_final_info", None)
        if final_obs_raw is None:
            return infos
        if isinstance(obs, dict):
            dense = {
                key: np.stack(
                    [
                        entry[key] if entry is not None else obs[key][i]
                        for i, entry in enumerate(final_obs_raw)
                    ]
                )
                for key in obs
            }
            infos["final_observation"] = _convert_obs(dense, self.device)
        else:
            dense = np.stack(
                [entry if entry is not None else obs[i] for i, entry in enumerate(final_obs_raw)]
            )
            infos["final_observation"] = _to_torch(dense, self.device)
        return infos

    def close(self) -> None:
        self._vec_env.close()


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

        return gym.make(env_id, **env_kwargs)

    return _env_fn


def make_mujoco_env(cfg: MujocoEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [_make_env_fn(cfg.env_id, cfg.env_kwargs) for _ in range(cfg.num_envs)]
    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.Env = _MujocoVecEnvAdapter(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper

        adapter = RewardScaleBiasWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
