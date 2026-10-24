"""Torch-native image frame stacking.

Two call conventions, both producing ``(N, T, H, W, C)`` from ``(N, H, W, C)``
image observations with vector state left single-frame:

- ``image_keys=None`` (default): the original ManiSkill-vector-env path.
  Derives ``image_keys`` from ``base_env._init_raw_obs`` (any ``rgb``/``depth``
  -prefixed key) and calls ``base_env.update_obs_space(...)`` to publish the
  stacked space -- unchanged from before ``image_keys`` was added.
- ``image_keys=<explicit sequence>``: for envs with no ``_init_raw_obs``/
  ``update_obs_space`` (e.g. ``FrankaRealEnv``, whose camera keys are
  user-configured and not necessarily ``rgb``/``depth``-prefixed). Derives the
  stacked space purely from ``single_observation_space`` -- deliberately never
  calls ``env.reset()`` to seed it, since construction-time side effects would
  physically move a real robot. Sets ``single_observation_space``/
  ``observation_space`` directly on this wrapper instance (mirroring
  ``RotvecObsWrapper``'s convention) since there's no ``update_obs_space`` to
  mutate, and forwards other attributes (``num_envs``, ``single_action_space``,
  ...) via ``__getattr__`` so wrappers stacked above this one still resolve
  them -- purely additive relative to the ManiSkill path, which never relied
  on ``__getattr__`` (it already worked through ``get_wrapper_attr()``/base
  ``gym.Wrapper``'s own ``action_space``/``observation_space`` properties).

Episode boundaries: ``reset()`` (full or partial ``options={"env_idx": ...}``)
fills the selected histories with ``frame_stack`` copies of the reset frame.
That covers ManiSkill, where this wrapper sits inside ``ManiSkillVectorEnv``,
which calls ``reset(env_idx)`` on done envs. Envs that auto-reset done envs
inside ``step()`` (gymnasium ``AutoresetMode.SAME_STEP``: the returned obs is
the new episode's first frame, the terminal frame is in
``infos["final_observation"]``) never call ``reset()`` per env, so pass
``autoreset_in_step=True``: ``step()`` then restarts each done env's history
from its new first frame and returns ``final_observation`` stacked as
``previous_history[1:] + terminal_frame`` (SB3 ``VecFrameStack``'s terminal
convention). ``final_observation`` is stacked for every env; consumers select
done envs with ``_final_observation``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


class ImageFrameStackWrapper(gym.Wrapper):
    """Stack image observations while leaving vector state single-frame.

    The wrapped environment is already batched, so image tensors use
    ``(N, H, W, C)`` and this wrapper returns ``(N, T, H, W, C)``. Partial
    resets replace history only for the selected environment indices.

    Set ``autoreset_in_step=True`` when the wrapped env auto-resets done envs
    inside ``step()`` and returns the post-reset obs plus
    ``infos["final_observation"]`` (see module docstring).
    """

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 3,
        image_keys: Optional[Sequence[str]] = None,
        autoreset_in_step: bool = False,
    ) -> None:
        if frame_stack < 2:
            raise ValueError("frame_stack must be at least 2")
        # Not super().__init__(env): gym.Wrapper.__init__ asserts
        # isinstance(env, gym.Env), which the generic (image_keys=...) path's
        # real callers violate -- it wraps an already-vectorized, batched env
        # post-TorchVectorEnvAdapter (a gymnasium.vector.VectorWrapper, not a
        # gym.Env), e.g. rl_garden.envs.ogbench.env.make_ogbench_env /
        # mujoco_warp.env.make_mujoco_warp_env. Replicates gym.Wrapper.__init__
        # verbatim minus that assertion; every attribute it sets is either
        # overwritten immediately below (observation_space) or accessed
        # through the inherited property with this None fallback
        # (action_space/metadata), so this is behavior-preserving for the
        # ManiSkill/IsaacLab path (a true gym.Env) too.
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None
        self.frame_stack = int(frame_stack)
        self.autoreset_in_step = bool(autoreset_in_step)
        self._frames: dict[str, torch.Tensor] = {}

        if image_keys is not None:
            self.image_keys = tuple(image_keys)
            if not self.image_keys:
                raise ValueError("ImageFrameStackWrapper requires image observations")
            self._update_obs_space_generic()
        else:
            self.image_keys = tuple(
                key
                for key in self.base_env._init_raw_obs
                if key.startswith(("rgb_", "depth_"))
            )
            if not self.image_keys:
                raise ValueError("ImageFrameStackWrapper requires image observations")
            initial = self._reset_frames(self.base_env._init_raw_obs, env_idx=None)
            self.base_env.update_obs_space(initial)

    @property
    def base_env(self):
        return self.env.unwrapped

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails -- doesn't shadow base
        # gym.Wrapper's own action_space/observation_space properties, and is
        # additive (not behavior-changing) for the ManiSkill path, which
        # never relied on this. Needed for the generic (image_keys=...) path
        # so attributes like num_envs/single_action_space still resolve for
        # any wrapper stacked above this one.
        return getattr(self.env, name)

    def _update_obs_space_generic(self) -> None:
        """Non-ManiSkill fallback: rewrite single_observation_space/
        observation_space directly from the space definition (no live obs
        sample, no reset() call). Load-bearing, not cosmetic -- e.g.
        ReplayBuffer sizes every tensor from observation_space at
        buffer-construction time, so an unstacked space here would
        shape-mismatch on the first add()."""
        base_space = self.env.single_observation_space
        new_spaces = dict(base_space.spaces)
        for key in self.image_keys:
            key_space = new_spaces[key]
            stacked_shape = (self.frame_stack, *key_space.shape)
            new_spaces[key] = spaces.Box(
                low=np.broadcast_to(key_space.low, stacked_shape),
                high=np.broadcast_to(key_space.high, stacked_shape),
                shape=stacked_shape,
                dtype=key_space.dtype,
            )
        self.single_observation_space = spaces.Dict(new_spaces)
        self.observation_space = batch_space(
            self.single_observation_space, self.env.num_envs
        )

    def _repeated(self, image: torch.Tensor) -> torch.Tensor:
        return image.unsqueeze(1).expand(-1, self.frame_stack, *image.shape[1:]).clone()

    def _output(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = dict(obs)
        output.update(self._frames)
        return output

    def _reset_frames(
        self,
        obs: dict[str, torch.Tensor],
        env_idx: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if env_idx is None or not self._frames:
            self._frames = {key: self._repeated(obs[key]) for key in self.image_keys}
            return self._output(obs)

        for key in self.image_keys:
            updated = self._frames[key].clone()
            updated[env_idx] = self._repeated(obs[key][env_idx])
            self._frames[key] = updated
        return self._output(obs)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        env_idx = None if options is None else options.get("env_idx")
        return self._reset_frames(obs, env_idx=env_idx), info

    def _stacked_final_observation(self, info: dict[str, Any]) -> dict[str, Any]:
        """Shallow copy of ``info`` whose ``final_observation`` image keys are
        ``previous_history[1:] + terminal_frame``; the inner env's dict is
        left untouched."""
        final_obs = info.get("final_observation")
        if not isinstance(final_obs, dict):
            return info
        stacked = dict(final_obs)
        for key in self.image_keys:
            stacked[key] = torch.cat(
                (self._frames[key][:, 1:], final_obs[key].unsqueeze(1)), dim=1
            )
        return {**info, "final_observation": stacked}

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frames = {
            key: torch.cat((self._frames[key][:, 1:], obs[key].unsqueeze(1)), dim=1)
            for key in self.image_keys
        }
        if self.autoreset_in_step:
            info = self._stacked_final_observation(info)
            # A fresh tensor: some envs (Isaac Lab) return persistent buffers.
            done = torch.logical_or(terminated, truncated)
            for key, rolled in frames.items():
                done_mask = done.to(device=rolled.device, dtype=torch.bool).view(
                    -1, *([1] * (rolled.dim() - 1))
                )
                frames[key] = torch.where(done_mask, obs[key].unsqueeze(1), rolled)
        self._frames = frames
        return self._output(obs), reward, terminated, truncated, info
