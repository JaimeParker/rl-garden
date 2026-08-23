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
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


class ImageFrameStackWrapper(gym.Wrapper):
    """Stack image observations while leaving vector state single-frame.

    The wrapped environment is already batched, so image tensors use
    ``(N, H, W, C)`` and this wrapper returns ``(N, T, H, W, C)``. Partial
    resets replace history only for the selected environment indices.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = 3,
        image_keys: Optional[Sequence[str]] = None,
    ) -> None:
        if frame_stack < 2:
            raise ValueError("frame_stack must be at least 2")
        super().__init__(env)
        self.frame_stack = int(frame_stack)
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
                if key.startswith(("rgb", "depth"))
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
        DictReplayBuffer sizes every tensor from observation_space at
        buffer-construction time, so an unstacked space here would
        shape-mismatch on the first add()."""
        base_space = self.env.single_observation_space
        new_spaces = dict(base_space.spaces)
        for key in self.image_keys:
            key_space = new_spaces[key]
            new_spaces[key] = spaces.Box(
                low=0,
                high=255,
                shape=(self.frame_stack, *key_space.shape),
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

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames = {
            key: torch.cat((self._frames[key][:, 1:], obs[key].unsqueeze(1)), dim=1)
            for key in self.image_keys
        }
        return self._output(obs), reward, terminated, truncated, info
