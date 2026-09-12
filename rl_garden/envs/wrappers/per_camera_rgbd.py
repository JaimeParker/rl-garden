"""Per-camera RGBD observation wrapper.

Mirrors :class:`mani_skill.utils.wrappers.flatten.FlattenRGBDObservationWrapper`
but keeps each requested camera's image as its own observation key
(``rgb_<camera_name>`` / ``depth_<camera_name>``) instead of channel-stacking
all cameras into a single ``rgb`` / ``depth`` tensor -- the observation
contract (``rl_garden.observations.schema``) has no bare ``rgb``/``depth``
keys, and fusing multiple cameras is the encoder's job, not the env's.

This matches hil-serl's ``EncodingWrapper`` style: each camera gets its own
3-channel encoder via :class:`CombinedExtractor` ``fusion_mode="per_key"``, so
ImageNet-pretrained ResNet stems load without channel-shape mismatches and
normalization stays correct per camera.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from rl_garden.observations.schema import ObservationContractError


class PerCameraRGBDWrapper(gym.ObservationWrapper):
    """Per-camera variant of ``FlattenRGBDObservationWrapper``.

    Args:
        env: The underlying ManiSkill env.
        rgb_cameras: Camera names to expose as ``rgb_<camera_name>``.
        depth_cameras: Camera names to expose as ``depth_<camera_name>``.
        state: Include the flat ``state`` key.

    Any name in ``rgb_cameras``/``depth_cameras`` not among the env's own
    sensors raises ``ObservationContractError`` listing the available ones.

    Output observation dict keys:
        * ``rgb_<camera_name>``: ``Box(shape=(H, W, 3), dtype=uint8)`` per requested camera
        * ``depth_<camera_name>``: ``Box(shape=(H, W, 1), dtype=float32)`` per requested camera
        * ``state``: flat agent/extra state, when ``state=True``
    """

    def __init__(
        self,
        env: gym.Env,
        rgb_cameras: tuple[str, ...] = (),
        depth_cameras: tuple[str, ...] = (),
        state: bool = True,
    ) -> None:
        # Late imports so importing rl_garden without mani_skill stays cheap.
        from mani_skill.envs.sapien_env import BaseEnv

        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.rgb_cameras = tuple(rgb_cameras)
        self.depth_cameras = tuple(depth_cameras)
        self.state = state

        available = set(self.base_env._init_raw_obs["sensor_data"].keys())
        unknown = (set(self.rgb_cameras) | set(self.depth_cameras)) - available
        if unknown:
            raise ObservationContractError(
                f"maniskill: unknown camera(s) {sorted(unknown)!r}; available "
                f"cameras: {sorted(available)}"
            )

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        from mani_skill.utils import common

        sensor_data = observation.pop("sensor_data")
        observation.pop("sensor_param", None)

        ret: dict[str, Any] = {}
        for cam_name in self.rgb_cameras:
            ret[f"rgb_{cam_name}"] = sensor_data[cam_name]["rgb"]
        for cam_name in self.depth_cameras:
            ret[f"depth_{cam_name}"] = sensor_data[cam_name]["depth"]

        if self.state:
            ret["state"] = common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        return ret
