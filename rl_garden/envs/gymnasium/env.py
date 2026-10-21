"""Generic ``gymnasium.make`` environment factory.

Vectorizes any registered, single-instance, numpy-based ``gymnasium.Env``
with ``SyncVectorEnv``/``AsyncVectorEnv`` (``autoreset_mode=SAME_STEP``) and
rl-garden's shared ``TorchVectorEnvAdapter`` -- the same shape as
``rl_garden.envs.mujoco.env`` / ``rl_garden.envs.ogbench.env``; see
``rl_garden.envs.vector_env`` for the ``final_observation``/``final_info``
contract.

Registration: the env thunk imports every ``cfg.import_modules`` entry
*inside* the thunk before ``gym.make``, so ``gymnasium.register`` side
effects also run in ``AsyncVectorEnv`` workers under a spawn/forkserver
start method, which don't inherit the parent's imports.

Observations: the env must emit rl-garden contract keys itself (``state``,
``state_<name>``, ``rgb_<cam>``, ``depth_<cam>``; checked by the registry
after construction). A bare 1-D float ``Box`` is the one shape auto-wrapped,
into ``Dict({"state": Box})``; any other bare ``Box`` is rejected.

Episode statistics: ``RecordEpisodeStatistics`` is applied unless the env's
wrapper chain already has one (a second instance asserts on its own info
key), since training/eval logging reads ``final_info["episode"]``.
"""
from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.envs.gymnasium.config import GymnasiumEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


class _SeededGymnasiumVecEnv(TorchVectorEnvAdapter):
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


def _import_user_modules(names: tuple[str, ...]) -> None:
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"gymnasium backend: cannot import {name!r} listed in "
                "--gymnasium.import_modules (is its package installed in this "
                f"Python environment?): {exc}"
            ) from exc


def _has_record_episode_statistics(env: gym.Env) -> bool:
    from gymnasium.wrappers import RecordEpisodeStatistics

    while isinstance(env, gym.Wrapper):
        if isinstance(env, RecordEpisodeStatistics):
            return True
        env = env.env
    return False


def _make_env_fn(env_id: str, env_kwargs: dict[str, Any], import_modules: tuple[str, ...]):
    def _env_fn():
        _import_user_modules(import_modules)

        from gymnasium.error import NameNotFound, NamespaceNotFound, VersionNotFound
        from gymnasium.wrappers import RecordEpisodeStatistics

        from rl_garden.envs.wrappers import DictStateObservationWrapper
        from rl_garden.observations.schema import ObservationContractError

        try:
            env = gym.make(env_id, **env_kwargs)
        except (NameNotFound, NamespaceNotFound, VersionNotFound) as exc:
            raise gym.error.Error(
                f"gymnasium backend: env_id {env_id!r} is not registered in this "
                "process. Register it with gymnasium.register(...) in a module "
                "listed in --gymnasium.import_modules, or write --env_id as "
                f"'my_pkg.module:{env_id}'. ({exc})"
            ) from exc
        space = env.observation_space
        if isinstance(space, spaces.Box):
            if len(space.shape) != 1 or not np.issubdtype(space.dtype, np.floating):
                raise ObservationContractError(
                    f"gymnasium backend: env_id {env_id!r} returns a bare Box "
                    f"observation of shape {space.shape!r} and dtype {space.dtype}; "
                    "only a 1-D float Box is auto-wrapped as 'state'. Return a Dict "
                    "observation keyed 'state' / 'state_<name>' / 'rgb_<cam>' / "
                    "'depth_<cam>' instead."
                )
            env = DictStateObservationWrapper(env)
        if not _has_record_episode_statistics(env):
            env = RecordEpisodeStatistics(env)
        return env

    return _env_fn


def make_gymnasium_env(cfg: GymnasiumEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [
        _make_env_fn(cfg.env_id, cfg.env_kwargs, tuple(cfg.import_modules))
        for _ in range(cfg.num_envs)
    ]
    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededGymnasiumVecEnv(
        vec_env, device=cfg.device, seed=cfg.seed
    )

    if cfg.frame_stack > 1:
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        adapter = ImageFrameStackWrapper(
            adapter, frame_stack=cfg.frame_stack, image_keys=cfg.image_keys
        )

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import (
            RewardScaleBiasVectorWrapper,
        )

        adapter = RewardScaleBiasVectorWrapper(
            adapter, scale=cfg.reward_scale, bias=cfg.reward_bias
        )

    return adapter
