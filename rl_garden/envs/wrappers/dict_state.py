"""Wraps a state-only ``Box`` observation into the observation contract's
``Dict({"state": Box})`` shape.

Used by every state-only env backend (mujoco, minari, d4rl_legacy, custom,
robomimic, and metaworld's state path) so they all emit
``spaces.Dict`` per ``rl_garden.observations.schema``, instead of each
backend hand-rolling the same one-key dict. Applied to the single,
pre-vectorization ``gym.Env`` inside each backend's ``_make_env_fn`` --
``TorchVectorEnvAdapter`` (``rl_garden.envs.vector_env``) already casts any
float64 array/space to float32 once vectorized
(``canonicalize_floating_observation_space`` + ``_to_torch``), so this
wrapper does not need to cast dtype itself.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.common.spaces import canonicalize_floating_observation_space
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


def require_state_only_observation(observation: ObservationConfig, *, backend: str) -> None:
    """Raise ``ObservationContractError`` unless ``observation`` asks only for
    state -- shared by every backend that has no camera/vision support at all
    (mujoco benchmark tasks, minari, d4rl_legacy, custom, robomimic)."""
    if observation.rgb or observation.depth:
        raise ObservationContractError(
            f"{backend} backend is state-only; requested cameras "
            f"rgb={observation.rgb!r} depth={observation.depth!r} cannot be honored"
        )
    if observation.image_size is not None:
        raise ObservationContractError(
            f"{backend} backend is state-only and cannot set image_size="
            f"{observation.image_size!r}"
        )
    if observation.frame_stack > 1:
        raise ObservationContractError(
            f"{backend} backend is state-only; frame_stack="
            f"{observation.frame_stack} only applies to image observations"
        )
    if not observation.state:
        raise ObservationContractError(
            f"{backend} backend is state-only; observation.state=False leaves "
            "nothing to observe"
        )


class DictStateObservationWrapper(gym.ObservationWrapper):
    """Wrap a flat ``Box`` observation into ``Dict({"state": Box})``."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError(
                "DictStateObservationWrapper requires a Box observation "
                f"space, got {type(base_space).__name__}"
            )
        self.observation_space = spaces.Dict({"state": base_space})
        # Already-vectorized envs (e.g. ManiSkill's GPU sim) expose a
        # separate `single_observation_space` (per-env) distinct from
        # `observation_space` (batched); gymnasium's Wrapper otherwise
        # proxies unset attributes straight through to the wrapped env via
        # __getattr__, leaving it stale (still the un-wrapped Box) since
        # only `observation_space` is overridden above. Mirror it here so
        # `env.single_observation_space` matches what `observation()` (and
        # therefore every rollout step) actually returns.
        base_single_space = getattr(env, "single_observation_space", None)
        if base_single_space is not None:
            if not isinstance(base_single_space, spaces.Box):
                raise TypeError(
                    "DictStateObservationWrapper requires a Box "
                    f"single_observation_space, got {type(base_single_space).__name__}"
                )
            self.single_observation_space = spaces.Dict({"state": base_single_space})

    def observation(self, observation):
        return {"state": observation}


class VectorizedDictStateWrapper:
    """Wraps an already-vectorized env whose ``single_observation_space`` is a
    bare ``Box`` into ``Dict({"state": Box})``, at the algorithm-construction
    boundary (``BaseAlgorithm.__init__``).

    Unlike ``DictStateObservationWrapper`` above (a ``gym.ObservationWrapper``
    applied to a single, pre-vectorization ``gym.Env`` inside a backend's
    ``_make_env_fn``), the env here is already vectorized: a real
    ``TorchVectorEnvAdapter`` (``rl_garden.envs.vector_env``), or a duck-typed
    vectorized test double. Neither is necessarily a
    ``gymnasium.Env``/``gymnasium.vector.VectorEnv`` instance -- test doubles
    are plain objects, and ``TorchVectorEnvAdapter`` is a
    ``gymnasium.vector.VectorWrapper``, not a ``gym.Env`` -- and both
    ``gym.Wrapper.__init__``/``VectorWrapper.__init__`` hard-assert
    ``isinstance(env, Env)``/``isinstance(env, VectorEnv)``. So this is a
    plain duck-typed proxy (attribute passthrough via ``__getattr__``), not a
    subclass of either.

    Observations here are already batched torch tensors of shape
    ``(num_envs, *shape)``; wrapping is just ``{"state": tensor}``.
    """

    def __init__(self, env) -> None:
        base_space = env.single_observation_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError(
                "VectorizedDictStateWrapper requires a Box single_observation_space, "
                f"got {type(base_space).__name__}"
            )
        self.env = env
        base_space = canonicalize_floating_observation_space(base_space)
        self.single_observation_space = spaces.Dict({"state": base_space})
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return {"state": obs}, info

    def step(self, actions):
        obs, reward, terminated, truncated, infos = self.env.step(actions)
        infos = dict(infos)
        final_obs = infos.get("final_observation")
        if final_obs is not None:
            infos["final_observation"] = {"state": final_obs}
        return {"state": obs}, reward, terminated, truncated, infos

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def __getattr__(self, name: str):
        # Only reached when normal attribute lookup (instance __dict__, class
        # dict) fails -- ``self.env`` is set in __init__ and found there
        # directly, so this does not recurse. Guarded anyway in case an
        # instance is accessed before __init__ runs (e.g. during unpickling).
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
