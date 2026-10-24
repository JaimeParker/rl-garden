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

Observations: the env must emit rl-garden contract keys (``state``,
``state_<name>``, ``rgb_<cam>``, ``depth_<cam>``; checked by the registry
after construction), itself or through ``obs_key_map``. A bare 1-D float
``Box`` is the one shape auto-wrapped, into ``Dict({"state": Box})``; any
other bare ``Box`` is rejected.

Episode statistics: ``RecordEpisodeStatistics`` is applied unless the env's
wrapper chain already has one (a second instance asserts on its own info
key), since training/eval logging reads ``final_info["episode"]``.

``obs_key_map`` renames the env's Dict observation keys to contract keys
once, at the vector level after the numpy-to-torch adapter, so it applies to
both the ``gym.make`` path and the ``vector_env_factory`` path.

``vector_env_factory`` (``"pkg.module:function"``) replaces ``gym.make`` +
``Sync/AsyncVectorEnv``: it is called as ``fn(num_envs, seed, device,
**env_kwargs)`` and must return a ``SAME_STEP`` vector env with a Dict
observation space that already reports ``final_info["episode"]``. A numpy
result goes through the adapter; a torch-native one
(``vector_env_returns_torch``) is used as is, only seeded.
"""
from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

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


class _SeededTorchVecEnv(gym.vector.VectorWrapper):
    """``reset()`` without a seed replays the configured seed, for a
    torch-native factory env that bypasses ``_SeededGymnasiumVecEnv``."""

    def __init__(self, vec_env: gym.vector.VectorEnv, seed: int) -> None:
        super().__init__(vec_env)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed if seed is not None else self._seed, options=options)


class _RenameObservationKeys(gym.vector.VectorWrapper):
    """Renames Dict observation keys (``key_map``: env key -> contract key)
    in the observation spaces, every ``reset``/``step`` observation, and
    ``infos["final_observation"]``; unmapped keys pass through."""

    def __init__(self, vec_env: gym.vector.VectorEnv, key_map: dict[str, str]) -> None:
        from rl_garden.observations.schema import ObservationContractError

        super().__init__(vec_env)
        single_space = vec_env.single_observation_space
        if not isinstance(single_space, spaces.Dict):
            raise ObservationContractError(
                "gymnasium backend: obs_key_map requires a Dict observation space, "
                f"got {type(single_space).__name__}"
            )
        keys = list(single_space.spaces)
        for src in key_map:
            if src not in single_space.spaces:
                raise ObservationContractError(
                    f"gymnasium backend: obs_key_map source {src!r} not in env "
                    f"observation keys {keys!r}"
                )
        renamed_keys = [key_map.get(key, key) for key in keys]
        if len(set(renamed_keys)) != len(renamed_keys):
            raise ObservationContractError(
                f"gymnasium backend: obs_key_map {key_map!r} renames a key onto another "
                f"env observation key {keys!r}"
            )
        self._key_map = key_map
        self.single_observation_space = spaces.Dict(self._rename(single_space.spaces))
        self.observation_space = batch_space(self.single_observation_space, vec_env.num_envs)

    def _rename(self, mapping: dict[str, Any]) -> dict[str, Any]:
        return {self._key_map.get(key, key): value for key, value in mapping.items()}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._rename(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        if isinstance(infos.get("final_observation"), dict):
            infos = {**infos, "final_observation": self._rename(infos["final_observation"])}
        return self._rename(obs), reward, terminated, truncated, infos


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


def _load_vector_env_factory(spec: str):
    module_name, _, function_name = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"gymnasium backend: cannot import module {module_name!r} of "
            f"--gymnasium.vector_env_factory {spec!r}: {exc}"
        ) from exc
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise AttributeError(
            f"gymnasium backend: vector_env_factory {spec!r}: module {module_name!r} "
            f"has no attribute {function_name!r}"
        ) from exc


def _make_factory_vec_env(cfg: GymnasiumEnvConfig) -> gym.vector.VectorEnv:
    from gymnasium.vector import AutoresetMode

    from rl_garden.observations.schema import ObservationContractError

    _import_user_modules(tuple(cfg.import_modules))
    factory = _load_vector_env_factory(cfg.vector_env_factory)
    vec_env = factory(cfg.num_envs, cfg.seed, cfg.device, **cfg.env_kwargs)
    if not isinstance(vec_env, gym.vector.VectorEnv):
        raise TypeError(
            f"gymnasium backend: vector_env_factory {cfg.vector_env_factory!r} must "
            f"return a gymnasium.vector.VectorEnv, got {type(vec_env).__name__}"
        )
    # Checked before any wrapping: the adapter forces this metadata entry.
    mode = vec_env.metadata.get("autoreset_mode")
    if mode != AutoresetMode.SAME_STEP:
        raise ValueError(
            "gymnasium backend: vector env must use AutoresetMode.SAME_STEP, "
            f"got metadata['autoreset_mode']={mode!r}"
        )
    space = vec_env.single_observation_space
    if not isinstance(space, spaces.Dict):
        raise ObservationContractError(
            "gymnasium backend: vector_env_factory must return a Dict observation "
            f"space (or use obs_key_map); got {space!r}"
        )
    if cfg.vector_env_returns_torch:
        return _SeededTorchVecEnv(vec_env, seed=cfg.seed)
    return _SeededGymnasiumVecEnv(vec_env, device=cfg.device, seed=cfg.seed)


def make_gymnasium_env(cfg: GymnasiumEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    if cfg.vector_env_factory:
        adapter = _make_factory_vec_env(cfg)
    else:
        env_fns = [
            _make_env_fn(cfg.env_id, cfg.env_kwargs, tuple(cfg.import_modules))
            for _ in range(cfg.num_envs)
        ]
        vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
        vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
        adapter = _SeededGymnasiumVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.obs_key_map:
        adapter = _RenameObservationKeys(adapter, cfg.obs_key_map)

    if cfg.frame_stack > 1:
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        adapter = ImageFrameStackWrapper(
            adapter,
            frame_stack=cfg.frame_stack,
            image_keys=cfg.image_keys,
            autoreset_in_step=True,
        )

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import (
            RewardScaleBiasVectorWrapper,
        )

        adapter = RewardScaleBiasVectorWrapper(
            adapter, scale=cfg.reward_scale, bias=cfg.reward_bias
        )

    return adapter
