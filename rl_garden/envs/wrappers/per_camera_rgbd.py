"""Per-camera RGBD observation wrapper.

Mirrors :class:`mani_skill.utils.wrappers.flatten.FlattenRGBDObservationWrapper`
but keeps each camera's image as its own observation key
(``rgb_<camera_name>`` / ``depth_<camera_name>``) instead of channel-stacking
all cameras into a single ``rgb`` / ``depth`` tensor.

This matches hil-serl's ``EncodingWrapper`` style: each camera gets its own
3-channel encoder via :class:`CombinedExtractor` ``fusion_mode="per_key"``, so
ImageNet-pretrained ResNet stems load without channel-shape mismatches and
normalization stays correct per camera.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym


class PerCameraRGBDWrapper(gym.ObservationWrapper):
    """Per-camera variant of ``FlattenRGBDObservationWrapper``.

    Args:
        env: The underlying ManiSkill env.
        rgb: Include per-camera RGB keys.
        depth: Include per-camera depth keys.
        state: Include the flat ``state`` key.

    Output observation dict keys:
        * ``rgb_<camera_name>``: ``Box(shape=(H, W, 3), dtype=uint8)`` per camera
        * ``depth_<camera_name>``: ``Box(shape=(H, W, 1), dtype=float32)`` per camera
        * ``state``: flat agent/extra state, when ``state=True``
    """

    def __init__(
        self,
        env: gym.Env,
        rgb: bool = True,
        depth: bool = True,
        state: bool = True,
    ) -> None:
        # Late imports so importing rl_garden without mani_skill stays cheap.
        from mani_skill.envs.sapien_env import BaseEnv

        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state

        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        if "depth" not in first_cam:
            self.include_depth = False

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        from mani_skill.utils import common

        sensor_data = observation.pop("sensor_data")
        observation.pop("sensor_param", None)

        ret: dict[str, Any] = {}
        for cam_name, cam_data in sensor_data.items():
            if self.include_rgb and "rgb" in cam_data:
                ret[f"rgb_{cam_name}"] = cam_data["rgb"]
            if self.include_depth and "depth" in cam_data:
                ret[f"depth_{cam_name}"] = cam_data["depth"]

        if self.include_state:
            ret["state"] = common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        return ret
