"""Real-Franka env backend — registered as ``"franka_real"``.

Unlike the sim backends, this one ignores ``req.num_envs``/``req.num_eval_envs``
(a real robot is always a batch of exactly one) and reuses the same env for
both train and eval -- there is no separate eval-time reconfiguration for
real hardware.
"""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class FrankaRealBackend(EnvBackend):
    config_field = "franka_real"

    @classmethod
    def _make_cfg(cls, req: EnvRequest):
        from rl_garden.envs.franka_real import FrankaRealEnvConfig

        fr = req.backend_config  # FrankaRealConfig or None
        env_kwargs = (
            json.loads(fr.env_kwargs_json) if fr is not None and fr.env_kwargs_json else {}
        )
        return FrankaRealEnvConfig(
            bridge_url=fr.bridge_url if fr is not None else "http://localhost:5000",
            action_scale=(
                (fr.action_scale_pos, fr.action_scale_rot)
                if fr is not None
                else (0.02, 0.1)
            ),
            gripper_threshold=fr.gripper_threshold if fr is not None else 0.5,
            max_episode_steps=fr.max_episode_steps if fr is not None else 100,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            **env_kwargs,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.franka_real import make_franka_real_env

        return make_franka_real_env(cls._make_cfg(req))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.franka_real import make_franka_real_env

        return make_franka_real_env(cls._make_cfg(req))


register_env_backend("franka_real", FrankaRealBackend)
