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
import numpy as np
import torch

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper

TeleopSource = Union[EETwistTeleOpWrapper, "SpaceMouseTeleOpWrapper"]  # noqa: F821


class _TeleopInterventionMixin:
    def _init_teleop_intervention(
        self,
        teleop: Optional[TeleopSource] = None,
        device: Literal["pico", "spacemouse"] = "pico",
        record_gripper: bool = True,
        teleop_init_timeout_s: float = 120.0,
        **teleop_kwargs: Any,
    ) -> None:
        self.record_gripper = bool(record_gripper)
        self._validate_configured_action_dim()
        if teleop is not None:
            self.teleop = teleop
        elif device == "pico":
            print(
                f"[teleop] initializing device=pico "
                f"record_gripper={self.record_gripper}",
                flush=True,
            )
            self.teleop = EETwistTeleOpWrapper(
                device="pico",
                init_timeout_s=teleop_init_timeout_s,
                **teleop_kwargs,
            )
        elif device == "spacemouse":
            from robot_infra.teleop.spacemouse import SpaceMouseTeleOpWrapper

            print(
                f"[teleop] initializing device=spacemouse "
                f"record_gripper={self.record_gripper}",
                flush=True,
            )
            self.teleop = SpaceMouseTeleOpWrapper(**teleop_kwargs)
        else:
            raise ValueError(f"device must be 'pico' or 'spacemouse', got {device!r}.")

    def __getattr__(self, name: str):
        # Gymnasium wrappers (>=1.0) no longer forward arbitrary attributes to
        # ``self.env`` -- but this repo's env-backend contract (num_envs,
        # single_observation_space, ...) relies on direct attribute access,
        # not ``get_wrapper_attr()``, so this wrapper must still be
        # transparent to algorithm code built against the unwrapped env.
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.teleop.reset()
        return self.env.reset(**kwargs)

    def _expected_action_dim(self) -> Optional[int]:
        space = getattr(self.env, "single_action_space", None)
        if space is None:
            return None
        return int(np.prod(space.shape))

    def _configured_action_dim(self) -> int:
        return 7 if self.record_gripper else 6

    def _validate_configured_action_dim(self) -> None:
        expected_dim = self._expected_action_dim()
        if expected_dim is None:
            return
        configured_dim = self._configured_action_dim()
        if configured_dim != expected_dim:
            raise ValueError(
                "Teleop intervention action dimension mismatch: "
                f"record_gripper={self.record_gripper} produces {configured_dim} dims, "
                f"but env.single_action_space expects {expected_dim}."
            )

    def _intervention_action(self, sample_action, device: torch.device) -> torch.Tensor:
        action_np = np.asarray(sample_action, dtype=np.float32)
        action_np = action_np[:7] if self.record_gripper else action_np[:6]
        expected_dim = self._expected_action_dim()
        if expected_dim is not None and action_np.size != expected_dim:
            raise ValueError(
                "Teleop intervention action dimension mismatch: "
                f"record_gripper={self.record_gripper} produced {action_np.size} dims, "
                f"but env.single_action_space expects {expected_dim}."
            )
        return torch.as_tensor(action_np, device=device, dtype=torch.float32).reshape(1, -1)

    def step(self, action: torch.Tensor):
        sample = self.teleop.poll()
        if not sample.intervened:
            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info["human_episode_end"] = sample.episode_end
            return obs, reward, terminated, truncated, info

        device = action.device if isinstance(action, torch.Tensor) else torch.device("cpu")
        human_action = self._intervention_action(sample.action, device)
        obs, reward, terminated, truncated, info = self.env.step(human_action)
        info = dict(info)
        info["intervene_action"] = human_action
        info["human_episode_end"] = sample.episode_end
        return obs, reward, terminated, truncated, info


class TeleopInterventionWrapper(_TeleopInterventionMixin, gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        teleop: Optional[TeleopSource] = None,
        device: Literal["pico", "spacemouse"] = "pico",
        record_gripper: bool = True,
        teleop_init_timeout_s: float = 120.0,
        **teleop_kwargs: Any,
    ) -> None:
        super().__init__(env)
        self._init_teleop_intervention(
            teleop=teleop,
            device=device,
            record_gripper=record_gripper,
            teleop_init_timeout_s=teleop_init_timeout_s,
            **teleop_kwargs,
        )


class TeleopInterventionVectorWrapper(_TeleopInterventionMixin, gym.vector.VectorWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        teleop: Optional[TeleopSource] = None,
        device: Literal["pico", "spacemouse"] = "pico",
        record_gripper: bool = True,
        teleop_init_timeout_s: float = 120.0,
        **teleop_kwargs: Any,
    ) -> None:
        super().__init__(env)
        self._init_teleop_intervention(
            teleop=teleop,
            device=device,
            record_gripper=record_gripper,
            teleop_init_timeout_s=teleop_init_timeout_s,
            **teleop_kwargs,
        )


class ManiSkillHITLInterventionVectorWrapper(TeleopInterventionVectorWrapper):
    """HITL teleop wrapper for ManiSkill's same-step autoreset vector envs."""

    def hitl_reset_after_done(self, next_obs: Any, info: dict[str, Any]) -> Any:
        del info
        self.teleop.reset()
        return next_obs
