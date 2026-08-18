"""Factory: vectorizes ``PointReachEnv`` and adapts it to rl-garden's torch
env contract.

This is the part a new environment author normally does NOT need to
change: vectorize any standard, numpy-based ``gymnasium.Env`` with
``gymnasium.vector.SyncVectorEnv`` (single process -- swap in
``AsyncVectorEnv`` only if your task needs process isolation, e.g. separate
OpenGL contexts for rendering, same reasoning as
``rl_garden.envs.mujoco.env``'s module docstring), then wrap the result in
``rl_garden.envs.vector_env.TorchVectorEnvAdapter`` to get
``num_envs``/``single_observation_space``/``single_action_space`` and
torch-tensor ``reset``/``step`` for free -- the same mechanism
``rl_garden.envs.mujoco.env.make_mujoco_env`` uses.
"""
from __future__ import annotations

from rl_garden.envs.custom.config import CustomEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


def _make_env_fn():
    def _env_fn():
        # Import here, not at module scope: this is what actually triggers
        # gym.register("PointReach-v0", ...) at the bottom of
        # point_reach_env.py. Under AsyncVectorEnv with a spawn start
        # method, worker processes don't inherit the parent's imports, so
        # registration must happen fresh in whichever process calls
        # gym.make() -- see rl_garden.envs.mujoco.env's module docstring
        # for the same reasoning.
        import gymnasium as gym

        import rl_garden.envs.custom.point_reach_env  # noqa: F401

        return gym.make("PointReach-v0")

    return _env_fn


def make_custom_env(cfg: CustomEnvConfig):
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    env_fns = [_make_env_fn() for _ in range(cfg.num_envs)]
    vec_env = SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter = TorchVectorEnvAdapter(vec_env, device=cfg.device)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(
            adapter, scale=cfg.reward_scale, bias=cfg.reward_bias
        )

    return adapter
