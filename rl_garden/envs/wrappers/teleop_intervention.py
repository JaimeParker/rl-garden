"""Human-in-the-loop intervention wrapper (the design doc's "SpaceMouse
intervention" component).

Built on rl-garden's own teleop bridge
(``robot_infra/teleop/utils/telo_op_control_twist.py``'s
``EETwistTeleOpWrapper``) rather than porting SERL's own SpaceMouse-specific
Python driver -- that existing utility already does exactly this job (a
ZMQ-fed device -> EE twist + gripper + an ``intervened`` flag) and already
supports the ``"pico"`` device end to end; its ``"spacemouse"`` device
option raises ``NotImplementedError`` (no HID parser exists for it yet) --
that's inherited, unchanged behavior, not something this wrapper adds or
papers over. The 7D ``TeleOpSample.action`` (6D EE twist + gripper) matches
``FrankaRealEnv``'s action convention exactly, so no conversion is needed
beyond dtype/device and adding the batch-of-1 leading dim.
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper


class TeleopInterventionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        teleop: Optional[EETwistTeleOpWrapper] = None,
        **teleop_kwargs: Any,
    ) -> None:
        super().__init__(env)
        self.teleop = teleop if teleop is not None else EETwistTeleOpWrapper(**teleop_kwargs)

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
