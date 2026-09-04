"""RLBench live env: builds ``rlbench.environment.Environment`` +
``TaskEnvironment`` directly (not ``gymnasium.make``) so the observation
dict can be flattened/renamed to rl-garden's conventions -- see
``rl_garden.buffers.rlbench_dataset``'s module docstring for why
``rlbench.gym.RLBenchEnv`` isn't reused as-is.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.buffers.rlbench_dataset import (
    build_default_rlbench_action_mode,
    build_rlbench_obs_config,
    build_rlbench_observation,
)
from rl_garden.envs.rlbench.config import RLBenchEnvConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


class RLBenchGymEnv(gym.Env):
    """Single-instance RLBench task, wrapped to gymnasium's 5-tuple API with
    a renamed/flattened Dict observation (``"state"`` + optional
    ``rgb_<camera>``/``depth_<camera>``)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, cfg: RLBenchEnvConfig) -> None:
        super().__init__()
        import rlbench  # noqa: F401 -- registers the tasks package machinery.
        from rlbench.environment import Environment
        from rlbench.utils import name_to_task_class

        self._obs_mode = cfg.obs_mode
        self._cameras = cfg.cameras

        obs_config = build_rlbench_obs_config(
            obs_mode=cfg.obs_mode, cameras=cfg.cameras, image_size=cfg.image_size
        )
        action_mode = build_default_rlbench_action_mode()
        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=cfg.headless,
            **cfg.env_kwargs,
        )
        self._env.launch()
        self._task_env = self._env.get_task(name_to_task_class(cfg.task_name))

        _, obs = self._task_env.reset()
        first_obs = self._extract_obs(obs)
        if isinstance(first_obs, dict):
            self.observation_space = spaces.Dict(
                {
                    key: spaces.Box(low=0, high=255, shape=value.shape, dtype=np.uint8)
                    if key.startswith("rgb")
                    else spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)
                    for key, value in first_obs.items()
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=first_obs.shape, dtype=np.float32
            )
        # MoveArmThenGripper (the composable action mode this integration
        # uses) does not implement action_bounds() -- only RLBench's own
        # preset action modes (e.g. JointPositionActionMode) do, confirmed
        # by a real install: calling it raises NotImplementedError. Bounds
        # are built here instead: +-1.0 rad/s for each arm (JointVelocity)
        # dim, matching Environment's own arm_max_velocity=1.0 default, and
        # [0, 1] for the single Discrete gripper dim (its own documented
        # >0.5-open/<0.5-closed contract).
        total_dim = self._env.action_shape[0]
        arm_dim = total_dim - 1  # Discrete gripper always contributes 1 dim.
        action_low = np.array([-1.0] * arm_dim + [0.0], dtype=np.float32)
        action_high = np.array([1.0] * arm_dim + [1.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=self._env.action_shape,
            dtype=np.float32,
        )

    def _extract_obs(self, obs: Any) -> np.ndarray | dict[str, np.ndarray]:
        return build_rlbench_observation(obs, obs_mode=self._obs_mode, cameras=self._cameras)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # RLBench's own rlbench.gym.RLBenchEnv does the same explicit
        # np.random.seed call (with the same TODO to use self.np_random
        # instead) -- variation sampling reads the global numpy RNG, not a
        # per-env one.
        np.random.seed(seed=seed)
        _, obs = self._task_env.reset()
        return self._extract_obs(obs), {}

    def step(self, action):
        obs, reward, terminated = self._task_env.step(action)
        return self._extract_obs(obs), float(reward), bool(terminated), False, {}

    def render(self):
        return None

    def close(self) -> None:
        self._env.shutdown()


class _SeededRLBenchVecEnv(TorchVectorEnvAdapter):
    """Defaults ``reset()`` to the configured seed when the caller doesn't
    pass one explicitly -- tiny duplicate of every other backend's own
    ``_Seeded*VecEnv`` (not shared: module-private in each source file, one
    trivial wrapper per call site is not worth a shared parent)."""

    def __init__(self, vec_env, device: str, seed: int) -> None:
        super().__init__(vec_env, device=device)
        self._default_seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=seed if seed is not None else self._default_seed, options=options)


def _make_env_fn(cfg: RLBenchEnvConfig):
    def _env_fn():
        env = RLBenchGymEnv(cfg)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return _env_fn


def make_rlbench_env(cfg: RLBenchEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    env_fns = [_make_env_fn(cfg) for _ in range(cfg.num_envs)]
    vector_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    vec_env = vector_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededRLBenchVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)
    return adapter
