"""Human-in-the-loop intervention wrapper (the design doc's "SpaceMouse
intervention" component).

Built on rl-garden's own teleop bridge
(``robot_infra/teleop/utils/telo_op_control_twist.py``'s
``EETwistTeleOpWrapper``) for the ``"pico"`` device, and on
``robot_infra/teleop/spacemouse/SpaceMouseTeleOpWrapper`` (ported from
HIL-SERL) for ``"spacemouse"`` -- the two devices work on fundamentally
different mechanisms (a ZMQ-fed absolute pose vs. a locally HID-polled rate
device), so they're two separate device-source classes rather than one
class force-fit to both, but both produce the same
``TeleOpSample``-shaped output this wrapper consumes. The 7D
``TeleOpSample.action`` (6D EE twist + gripper) matches ``FrankaRealEnv``'s
action convention exactly, so no conversion is needed beyond dtype/device
and adding the batch-of-1 leading dim.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

import gymnasium as gym
import torch

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper

TeleopSource = Union[EETwistTeleOpWrapper, "SpaceMouseTeleOpWrapper"]  # noqa: F821


class TeleopInterventionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        teleop: Optional[TeleopSource] = None,
        device: Literal["pico", "spacemouse"] = "pico",
        **teleop_kwargs: Any,
    ) -> None:
        super().__init__(env)
        if teleop is not None:
            self.teleop = teleop
        elif device == "pico":
            self.teleop = EETwistTeleOpWrapper(device="pico", **teleop_kwargs)
        elif device == "spacemouse":
            from robot_infra.teleop.spacemouse import SpaceMouseTeleOpWrapper

            self.teleop = SpaceMouseTeleOpWrapper(**teleop_kwargs)
        else:
            raise ValueError(f"device must be 'pico' or 'spacemouse', got {device!r}.")

    def __getattr__(self, name: str):
        # gymnasium.Wrapper (>=1.0) no longer forwards arbitrary attributes to
        # ``self.env`` -- but this repo's env-backend contract (num_envs,
        # single_observation_space, ...) relies on direct attribute access,
        # not ``get_wrapper_attr()``, so this wrapper must still be
        # transparent to algorithm code built against the unwrapped env.
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.teleop.reset()
        return self.env.reset(**kwargs)

    def step(self, action: torch.Tensor):
        sample = self.teleop.poll()
        if not sample.intervened:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info["human_episode_end"] = sample.episode_end
            return obs, reward, terminated, truncated, info

        device = action.device if isinstance(action, torch.Tensor) else torch.device("cpu")
        human_action = torch.as_tensor(sample.action, device=device, dtype=torch.float32).unsqueeze(0)
        obs, reward, terminated, truncated, info = self.env.step(human_action)
        info = dict(info)
        info["intervene_action"] = human_action
        info["human_episode_end"] = sample.episode_end
        return obs, reward, terminated, truncated, info
