"""Meta-World environment factory.

Meta-World (Farama-Foundation/Metaworld) is a single-agent MuJoCo
manipulation benchmark. Two shapes of env id are supported, both driven
entirely through ``gymnasium.make`` (``import metaworld`` registers every
env id as an import side-effect, same as ``import ogbench``):

- A single task name (e.g. ``"reach-v3"``, any of ``metaworld.MT1
  .ENV_NAMES``): ``gym.make("Meta-World/MT1", env_name=...)`` returns one
  fully-wrapped, *non-vectorized* ``gym.Env`` (goal/object pose randomized
  automatically on every reset by Meta-World's own
  ``RandomTaskSelectWrapper``) -- Meta-World never vectorizes this branch
  on its own (confirmed by reading ``metaworld.make_mt_envs``: the entry
  point's own ``vector_strategy``/``num_envs`` kwargs are silently dropped
  for a single task name), so this module vectorizes it itself, the same
  shape as ``rl_garden.envs.ogbench.env``.
- ``"MT10"``/``"MT50"``: ``gym.make("Meta-World/MT10"|"MT50",
  vector_strategy=..., use_one_hot=...)`` already returns a real
  ``gym.vector.VectorEnv`` -- one sub-env per task, with the task count
  fixed upstream at 10/50 respectively. ``cfg.num_envs`` is meaningless for
  this branch (Meta-World silently ignores it too) and is not consulted.

State observations are the default; ``obs_mode == "rgb"`` adds a single
fixed-camera rgb+depth pair on top of the state vector, single-task env ids
only (see ``_MetaWorldVisionWrapper`` below) -- Meta-World has no
first-class pixel-obs env id of its own.

Meta-World's own ``RecordEpisodeStatistics`` (part of the wrapper stack
``gym.make("Meta-World/MT1", ...)`` applies internally) only ever puts
``{"r", "l", "t"}`` into ``info["episode"]`` -- never the ``info["success"]``
scalar Meta-World's own ``AutoTerminateOnSuccessWrapper`` also sets every
step. rl-garden's shared eval-metric extraction
(``rl_garden.algorithms.offline._append_episode_metrics``) only ever reads
keys out of ``info["episode"]``, so without a fix ``success_at_end`` is
always ``nan`` for this backend -- confirmed with a real eval run before
adding ``_MetaWorldEpisodeMetrics`` below. `robomimic`'s own
``_RobomimicEpisodeMetrics`` (``rl_garden/envs/robomimic/env.py``) solves the
identical problem the identical way for that backend's own success key
(``info["is_success"]["task"]``) -- mirrored here for
``info["success"]``. Only wired into the single-task branch: Meta-World's
own ``make_mt_envs`` builds MT10/MT50's sub-envs internally with no
per-sub-env hook this module can wrap, so ``success_at_end`` stays
unavailable for that branch (documented as a known limitation, not silently
dropped). The same MT10/MT50 limitation applies to vision below.
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from rl_garden.envs.metaworld.config import MetaWorldEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter

_MULTI_TASK_ENV_IDS = ("MT10", "MT50")


class _MetaWorldEpisodeMetrics(gym.Wrapper):
    """Attach ``success_at_end`` to terminal episode statistics -- see
    module docstring."""

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and "episode" in info:
            info = dict(info)
            episode = dict(info["episode"])
            episode["success_at_end"] = float(info.get("success", 0.0))
            info["episode"] = episode
        return obs, reward, terminated, truncated, info


class _MetaWorldVisionWrapper(gym.Wrapper):
    """Adds a single fixed-camera rgb+depth pair to the observation, turning
    the flat state ``Box`` into a ``Dict`` (``"state"`` plus
    ``rgb_<camera>``/``depth_<camera>``) -- rl-garden's standard
    state-is-flat-Box/vision-is-Dict convention (matches
    ``rl_garden.buffers.rlbench_dataset.build_rlbench_observation``).

    Built the same way ``rl_garden.envs.mujoco.custom_mujoco_env
    .CustomMujocoEnv._render_cameras()`` does for hand-authored MuJoCo
    tasks -- a ``MujocoRenderer`` bound to ``env.unwrapped.model``/``.data``
    and a single camera, since Meta-World's own ``SawyerXYZEnv`` is (like
    any ``CustomMujocoEnv`` subclass) a ``gymnasium.envs.mujoco.MujocoEnv``
    and exposes the same ``model``/``data`` attributes -- just applied as a
    wrapper around an existing env instead of built into a subclass, since
    Meta-World's env classes aren't ours to subclass.

    Depth is raw MuJoCo NDC (``mujoco.mjr_readPixels``'s buffer via
    ``OffScreenViewer.render()``), **not** linear/metric depth -- same
    caveat ``custom_mujoco_env.py`` documents (there empirically in the
    ``[0.987, 1.0]`` range for a mostly-background scene). No linearization
    applied here either.
    """

    def __init__(self, env: gym.Env, *, camera: str, image_size: tuple[int, int]) -> None:
        super().__init__(env)
        self._camera = camera
        height, width = image_size
        self._renderer = MujocoRenderer(env.unwrapped.model, env.unwrapped.data, width=width, height=height, camera_name=camera)
        state_space = env.observation_space
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=state_space.low.astype(np.float32),
                    high=state_space.high.astype(np.float32),
                    dtype=np.float32,
                ),
                f"rgb_{camera}": spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
                f"depth_{camera}": spaces.Box(-np.inf, np.inf, (height, width, 1), dtype=np.float32),
            }
        )

    def _augment(self, state_obs: np.ndarray) -> dict[str, np.ndarray]:
        rgb = self._renderer.render("rgb_array")
        depth = self._renderer.render("depth_array")[..., None].astype(np.float32)
        return {
            "state": state_obs.astype(np.float32),
            f"rgb_{self._camera}": rgb,
            f"depth_{self._camera}": depth,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def close(self):
        self._renderer.close()
        super().close()


class _SeededMetaWorldVecEnv(TorchVectorEnvAdapter):
    """Defaults ``reset()`` to the configured ``seed`` when the caller omits
    one, so repeated evaluation rounds replay the same fixed episodes
    (mirrors ``rl_garden.envs.ogbench.env._SeededOGBenchVecEnv`` --
    duplicated rather than shared since it's a module-private helper, not a
    subclassing hook)."""

    def __init__(self, vec_env: gym.vector.VectorEnv, device: str, seed: int) -> None:
        super().__init__(vec_env, device)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=seed if seed is not None else self._seed, options=options)


def _make_single_task_env_fn(env_id: str, seed: int, *, obs_mode: str, camera: str, image_size: tuple[int, int]):
    def _env_fn():
        # Default to the GPU-accelerated headless backend before any
        # rendering happens (relevant when obs_mode == "rgb"); setdefault so
        # an explicit caller override still wins (same as the ogbench
        # backend's identical pattern).
        os.environ.setdefault("MUJOCO_GL", "egl")
        import metaworld  # noqa: F401 -- registers "Meta-World/*" gym ids.

        env: gym.Env = _MetaWorldEpisodeMetrics(gym.make("Meta-World/MT1", env_name=env_id, seed=seed))
        if obs_mode != "state":
            env = _MetaWorldVisionWrapper(env, camera=camera, image_size=image_size)
        return env

    return _env_fn


def make_metaworld_env(cfg: MetaWorldEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    import metaworld  # noqa: F401 -- registers "Meta-World/*" gym ids.

    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv

    if cfg.env_id in _MULTI_TASK_ENV_IDS:
        if cfg.obs_mode != "state":
            raise ValueError(
                f"metaworld backend: obs_mode={cfg.obs_mode!r} is not supported for "
                f"env_id={cfg.env_id!r} -- MT10/MT50 build their sub-envs internally with "
                "no per-env construction hook to attach a camera renderer to. Use a "
                "single-task env_id (e.g. 'reach-v3') for vision observations."
            )
        # MT10/MT50 are registered with a `vector_entry_point` only (no
        # `entry_point`) -- gym.make_vec, not gym.make, matches Meta-World's
        # own README usage example for these ids.
        vec_env: gym.vector.VectorEnv = gym.make_vec(
            f"Meta-World/{cfg.env_id}",
            vector_strategy=cfg.vectorization,
            seed=cfg.seed,
            use_one_hot=cfg.use_one_hot,
        )
    else:
        env_fns = [
            _make_single_task_env_fn(
                cfg.env_id, cfg.seed + i, obs_mode=cfg.obs_mode, camera=cfg.camera, image_size=cfg.image_size
            )
            for i in range(cfg.num_envs)
        ]
        vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)

    adapter: gym.vector.VectorEnv = _SeededMetaWorldVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
