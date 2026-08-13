"""IsaacLab environment factory.

Wraps a raw IsaacLab env (``ManagerBasedRLEnv`` or ``DirectRLEnv``, native
``isaaclab_tasks`` task or an ``RLGardenDirectRLEnv``-scaffold task) with
``_IsaacLabVecEnvAdapter`` so it exposes the same shape the rest of
``rl_garden`` targets: ``num_envs`` / ``single_observation_space`` /
``single_action_space``, and ``step``/``reset`` returning GPU tensors plus
the standard Gymnasium ``(obs, reward, terminated, truncated, info)`` tuple.

Observation extraction:
- ``obs_mode="state"``: looks for a ``"state"`` key (rl-garden scaffold
  convention) first, falling back to ``"policy"`` (every native IsaacLab
  task's own convention -- both ``ManagerBasedRLEnv`` and ``DirectRLEnv``
  tasks use it).
- ``obs_mode`` in ``("rgb", "rgbd")``: requires rl-garden's own cross-backend
  key convention directly (``"rgb"``/``"depth"``/``"rgb_<cam>"``/
  ``"depth_<cam>"``, plus ``"state"`` if the task returns one) -- only
  ``RLGardenDirectRLEnv``-scaffold tasks are expected to satisfy this;
  native single-``"policy"``-image tasks (e.g. ``Isaac-Cartpole-RGB-Camera-
  Direct-v0``) aren't supported by this backend.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from rl_garden.envs.isaaclab.config import IsaacLabEnvConfig


def _box_from_tensor(tensor: torch.Tensor) -> gym.spaces.Box:
    if tensor.dtype == torch.uint8:
        return gym.spaces.Box(low=0, high=255, shape=tuple(tensor.shape[1:]), dtype=np.uint8)
    return gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=tuple(tensor.shape[1:]),
        dtype=np.float32,
    )


class _IsaacLabVecEnvAdapter(gym.Env):
    """Adapts an IsaacLab env to the rl-garden env shape (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, env: Any, seed: int, obs_mode: str = "state") -> None:
        self._env = env
        self._seed = seed
        self.obs_mode = obs_mode
        # gym.make() wraps the raw env in OrderEnforcing, which rejects
        # attribute access before the first reset() -- go through .unwrapped
        # for env-level attributes that aren't part of the standard gym.Env
        # surface (num_envs, device).
        unwrapped = env.unwrapped
        self.num_envs = unwrapped.num_envs
        self.device = unwrapped.device

        obs_dict, _ = env.reset(seed=seed)
        self._last_obs = self._extract_obs(obs_dict)
        # Duck-typed attribute expected by wrappers like ImageFrameStackWrapper
        # on env.unwrapped (mirrors mani_skill.envs.sapien_env.BaseEnv).
        self._init_raw_obs = self._last_obs
        self.single_observation_space = self._build_observation_space(self._last_obs)

        # IsaacLab's action_space is already batched as (num_envs, action_dim),
        # unlike Gymnasium's vector-env convention where single_action_space
        # describes one env -- slice off the leading num_envs dimension.
        action_space = unwrapped.action_space
        self.single_action_space = gym.spaces.Box(
            low=action_space.low[0],
            high=action_space.high[0],
            shape=action_space.shape[1:],
            dtype=action_space.dtype,
        )

    def _extract_obs(self, obs_dict: dict):
        if self.obs_mode == "state":
            if "state" in obs_dict:
                return obs_dict["state"]
            if "policy" in obs_dict:
                return obs_dict["policy"]
            raise ValueError(
                "IsaacLab backend (obs_mode='state') requires a 'state' or "
                f"'policy' observation key, got keys {sorted(obs_dict)!r}."
            )
        image_keys = [key for key in obs_dict if key.startswith(("rgb", "depth"))]
        if not image_keys:
            raise ValueError(
                f"IsaacLab backend (obs_mode={self.obs_mode!r}) requires "
                "'rgb'/'depth'-prefixed observation keys (rl-garden's own "
                "cross-backend convention -- see RLGardenDirectRLEnv's "
                f"docstring), got keys {sorted(obs_dict)!r}."
            )
        out = {key: obs_dict[key] for key in image_keys}
        if "state" in obs_dict:
            out["state"] = obs_dict["state"]
        return out

    def _build_observation_space(self, obs):
        if isinstance(obs, dict):
            return gym.spaces.Dict({key: _box_from_tensor(value) for key, value in obs.items()})
        return _box_from_tensor(obs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs_dict, info = self._env.reset(seed=seed if seed is not None else self._seed)
        self._last_obs = self._extract_obs(obs_dict)
        return self._last_obs, info

    def step(self, actions: torch.Tensor):
        obs_dict, reward, terminated, truncated, info = self._env.step(actions)
        self._last_obs = self._extract_obs(obs_dict)
        return self._last_obs, reward, terminated, truncated, info

    def update_obs_space(self, obs) -> None:
        """Duck-typed hook expected on ``env.unwrapped`` by wrappers like
        ``ImageFrameStackWrapper`` (mirrors
        ``mani_skill.envs.sapien_env.BaseEnv.update_obs_space``)."""
        self._init_raw_obs = obs
        self.single_observation_space = self._build_observation_space(obs)

    def post_update_sync(self) -> None:
        """Duck-typed hook ``OnPolicyAlgorithm.learn()`` calls once per
        rollout+update cycle, right after the update phase and before the
        next rollout's first ``step()``.

        Mitigation (not a confirmed full fix) for a render-pipeline stall
        observed on 6017-nofwd: Kit's camera-rendering pipeline can hang
        indefinitely on a render call that follows a sustained PyTorch CUDA
        compute burst (e.g. a PPO update's many backward passes). A single
        ``torch.cuda.synchronize()`` here measurably reduces how often this
        happens (state-only tasks never need it; observed several camera-
        task runs get through many more rollout+update cycles with this than
        without) but did NOT reliably eliminate the stall across repeated
        runs in testing -- it looks like a timing-sensitive race rather than
        something a single sync deterministically resolves. Syncing before
        every ``step()`` instead of once per cycle made it worse (reintroduced
        the hang reliably), so keep this placement (once per cycle) if
        investigating further.
        """
        if self.obs_mode != "state":
            torch.cuda.synchronize()

    def close(self) -> None:
        self._env.close()


def make_isaaclab_env(cfg: IsaacLabEnvConfig):
    """Build an IsaacLab env according to ``cfg``."""
    from rl_garden.envs.isaaclab.app import get_or_launch_app

    get_or_launch_app(cfg.headless, cfg.sim_device, enable_cameras=(cfg.obs_mode != "state"))

    import isaaclab_tasks  # noqa: F401  (registers native gym ids)

    # Must come after get_or_launch_app(): registering rl-garden's own tasks
    # imports RLGardenDirectRLEnv, which subclasses isaaclab.envs.DirectRLEnv
    # -- IsaacLab requires Kit to already be running before any isaaclab.*
    # subclass can be imported.
    import rl_garden.envs.isaaclab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(cfg.env_id, device=cfg.sim_device, num_envs=cfg.num_envs)
    for key, value in cfg.env_kwargs.items():
        setattr(env_cfg, key, value)

    env = gym.make(cfg.env_id, cfg=env_cfg)
    adapter = _IsaacLabVecEnvAdapter(env, seed=cfg.seed, obs_mode=cfg.obs_mode)

    if cfg.frame_stack > 1:
        if cfg.obs_mode == "state":
            raise ValueError("frame_stack > 1 requires a visual observation mode")
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        wrapped = ImageFrameStackWrapper(adapter, frame_stack=cfg.frame_stack)
        # ImageFrameStackWrapper doesn't forward arbitrary attributes -- copy
        # the rl-garden env-shape attributes onto the outermost object.
        # single_observation_space is already the post-frame-stack space at
        # this point (ImageFrameStackWrapper.__init__ calls update_obs_space
        # on the adapter before returning).
        wrapped.num_envs = adapter.num_envs
        wrapped.device = adapter.device
        wrapped.single_action_space = adapter.single_action_space
        wrapped.single_observation_space = adapter.single_observation_space
        wrapped.post_update_sync = adapter.post_update_sync
        return wrapped

    return adapter
