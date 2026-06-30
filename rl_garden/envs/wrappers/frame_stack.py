"""Torch-native image frame stacking for ManiSkill vector environments."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch


class ImageFrameStackWrapper(gym.Wrapper):
    """Stack image observations while leaving vector state single-frame.

    The wrapped ManiSkill environment is already batched, so image tensors use
    ``(N, H, W, C)`` and this wrapper returns ``(N, T, H, W, C)``. Partial
    resets replace history only for the selected environment indices.
    """

    def __init__(self, env: gym.Env, frame_stack: int = 3) -> None:
        if frame_stack < 2:
            raise ValueError("frame_stack must be at least 2")
        super().__init__(env)
        self.frame_stack = int(frame_stack)
        self.image_keys = tuple(
            key
            for key in self.base_env._init_raw_obs
            if key.startswith(("rgb", "depth"))
        )
        if not self.image_keys:
            raise ValueError("ImageFrameStackWrapper requires image observations")

        self._frames: dict[str, torch.Tensor] = {}
        initial = self._reset_frames(self.base_env._init_raw_obs, env_idx=None)
        self.base_env.update_obs_space(initial)

    @property
    def base_env(self):
        return self.env.unwrapped

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
