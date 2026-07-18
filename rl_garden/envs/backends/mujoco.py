"""MuJoCo env backend — registered as ``"mujoco"``.

Wraps either Gymnasium's registered MuJoCo benchmark tasks or rl-garden's own
custom tasks (``rl_garden.envs.mujoco.custom_mujoco_env.CustomMujocoEnv``);
see ``rl_garden.envs.mujoco.env`` module docstring for the SAME_STEP autoreset
/ ``final_observation`` contract and the sync/async vectorization choice.
Both train and eval envs are fully supported (Gymnasium's vector envs have no
single-process-instance constraint, unlike IsaacLab's Kit app).
"""
from __future__ import annotations

import json

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)


class MujocoBackend(EnvBackend):
    config_field = "mujoco"

    @classmethod
    def _make_cfg(cls, req: EnvRequest, *, is_eval: bool):
        from rl_garden.envs.mujoco import MujocoEnvConfig

        mj = req.backend_config  # MujocoConfig or None
        env_kwargs = (
            json.loads(mj.env_kwargs_json) if mj is not None and mj.env_kwargs_json else {}
        )
        return MujocoEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            device=mj.device if mj is not None else "cpu",
            env_kwargs=env_kwargs,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
            vectorization=mj.vectorization if mj is not None else "sync",
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco import make_mujoco_env

        return make_mujoco_env(cls._make_cfg(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from rl_garden.envs.mujoco import make_mujoco_env

        return make_mujoco_env(cls._make_cfg(req, is_eval=True))


register_env_backend("mujoco", MujocoBackend)
