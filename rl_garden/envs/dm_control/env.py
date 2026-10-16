"""dm_control environment factory.

Wraps one ``dm_control.suite`` task (standard suite or NCRL's custom tasks,
see ``custom_tasks``) as a numpy ``gymnasium.Env`` and vectorizes it the
same way ``rl_garden.envs.metaworld.env`` does a single-task env:
``SyncVectorEnv``/``AsyncVectorEnv`` (SAME_STEP autoreset) behind
``TorchVectorEnvAdapter``.

Per-env semantics mirror NCRL's ``envs.make`` wrapper stack
(``HardTaskWrapper`` -> ``TimeLimitWrapper`` -> ``ActionRepeatWrapper``):

- the hard-task reward transform applies per physics frame;
- the episode is truncated after ``episode_frames`` physics frames (1000),
  i.e. ``episode_frames / action_repeat`` policy steps;
- ``action_repeat`` frames are stepped per policy step, rewards summed,
  stopping early at the end of an episode.

Observations follow rl-garden's key contract: ``"state"`` is the
dm_control observation dict flattened in its fixed key order
(``environment_kwargs=dict(flat_observation=True)``, identical to NCRL's
state vector) cast to float32; ``"rgb_default"`` is a uint8 ``(H, W, 3)``
render (HWC, like every other rl-garden pixel backend -- encoders permute to
NCHW themselves) of camera ``camera_id``.

``dm_control`` is imported only when an env is constructed. Pixel rendering
needs a headless GL backend on servers; this module never sets
``MUJOCO_GL`` itself -- it raises with a hint when it is missing.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_garden.envs.dm_control.config import DMControlEnvConfig
from rl_garden.envs.dm_control.tasks import parse_task_id
from rl_garden.envs.vector_env import TorchVectorEnvAdapter
from rl_garden.observations.schema import ObservationContractError

# The single camera vocabulary entry: renders MuJoCo camera ``camera_id``.
DEFAULT_CAMERA_NAME = "default"

# NCRL finetune.py: cheetah tasks use 0.2, every other domain 0.6.
_HARD_REWARD_THRESHOLDS = {"cheetah": 0.2}
_HARD_DEFAULT_REWARD_THRESHOLD = 0.6
HARD_ACTION_PENALTY = 0.1

_INSTALL_HINT = (
    "The dm_control env backend needs the optional dm_control dependencies: "
    "pip install -e '.[dm_control]'"
)
_GL_HINT = (
    "Pixel observations from dm_control need an OpenGL backend. On a headless "
    "server set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU) before starting "
    "Python."
)


def default_camera_id(domain: str) -> int:
    return 2 if domain == "quadruped" else 0


def hard_reward_threshold(domain: str) -> float:
    return _HARD_REWARD_THRESHOLDS.get(domain, _HARD_DEFAULT_REWARD_THRESHOLD)


def hard_task_reward(
    reward: float,
    action: np.ndarray,
    *,
    reward_threshold: float,
    action_penalty: float = HARD_ACTION_PENALTY,
) -> float:
    """NCRL ``HardTaskWrapper``: zero rewards below ``reward_threshold``,
    then subtract ``action_penalty * ||action||``."""
    if reward < reward_threshold:
        reward = 0.0
    return float(reward) - action_penalty * float(np.linalg.norm(action))


def validate_cameras(rgb_cameras: tuple[str, ...]) -> None:
    unknown = [cam for cam in rgb_cameras if cam != DEFAULT_CAMERA_NAME]
    if unknown:
        raise ObservationContractError(
            f"dm_control: unknown camera(s) {unknown!r}; the only camera name is "
            f"{DEFAULT_CAMERA_NAME!r} (renders MuJoCo camera --dm_control.camera_id, "
            "default 2 for quadruped and 0 otherwise)."
        )


def _require_gl_backend() -> None:
    if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
        raise RuntimeError(f"MUJOCO_GL is not set and no DISPLAY is available. {_GL_HINT}")


class DMControlGymEnv(gym.Env):
    """One dm_control task as a numpy ``gymnasium.Env`` with a Dict
    observation and a ``Box(-1, 1)`` action space (rescaled to the task's
    action spec bounds)."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        env_id: str,
        *,
        seed: int,
        action_repeat: int = 2,
        episode_frames: int = 1000,
        hard: bool = False,
        camera_id: Optional[int] = None,
        state: bool = True,
        rgb_cameras: tuple[str, ...] = (),
        image_size: tuple[int, int] = (64, 64),
    ) -> None:
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")
        validate_cameras(rgb_cameras)
        domain, task, hard_suffix = parse_task_id(env_id)
        if rgb_cameras:
            _require_gl_backend()
        try:
            from dm_control import suite
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        import rl_garden.envs.dm_control.custom_tasks  # noqa: F401 -- registers NCRL tasks.

        self._env = suite.load(
            domain,
            task,
            task_kwargs={"random": seed},
            environment_kwargs={"flat_observation": True},
            visualize_reward=False,
        )
        self._action_repeat = action_repeat
        self._episode_frames = episode_frames
        self._hard = hard or hard_suffix
        self._reward_threshold = hard_reward_threshold(domain)
        self._camera_id = default_camera_id(domain) if camera_id is None else camera_id
        self._state = state
        self._rgb_cameras = tuple(rgb_cameras)
        self._image_size = tuple(image_size)
        self._frames = 0

        action_spec = self._env.action_spec()
        self._action_low = action_spec.minimum.astype(np.float64)
        self._action_high = action_spec.maximum.astype(np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, action_spec.shape, dtype=np.float32)

        height, width = self._image_size
        obs_spaces: dict[str, spaces.Box] = {}
        for camera in self._rgb_cameras:
            obs_spaces[f"rgb_{camera}"] = spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        if state:
            state_dim = self._env.observation_spec()["observations"].shape
            obs_spaces["state"] = spaces.Box(-np.inf, np.inf, state_dim, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    def _render(self) -> np.ndarray:
        height, width = self._image_size
        try:
            return self._env.physics.render(height=height, width=width, camera_id=self._camera_id)
        except Exception as exc:  # GL context/backend failures surface here.
            raise RuntimeError(f"dm_control render failed ({exc!r}). {_GL_HINT}") from exc

    def _observe(self, time_step) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}
        if self._rgb_cameras:
            frame = self._render()
            for camera in self._rgb_cameras:
                obs[f"rgb_{camera}"] = frame
        if self._state:
            obs["state"] = np.asarray(time_step.observation["observations"], dtype=np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.task.random.seed(seed)
        self._frames = 0
        return self._observe(self._env.reset()), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        physics_action = self._action_low + (action + 1.0) * 0.5 * (self._action_high - self._action_low)
        reward = 0.0
        end_of_episode = False
        for _ in range(self._action_repeat):
            time_step = self._env.step(physics_action)
            self._frames += 1
            frame_reward = time_step.reward or 0.0
            if self._hard:
                frame_reward = hard_task_reward(
                    frame_reward, action, reward_threshold=self._reward_threshold
                )
            reward += frame_reward
            end_of_episode = time_step.last() or self._frames >= self._episode_frames
            if end_of_episode:
                break
        # dm_control signals a true termination by ending with discount 0;
        # the suite tasks used here never do, so episodes end by time limit.
        terminated = bool(time_step.last() and time_step.discount == 0.0)
        truncated = bool(end_of_episode and not terminated)
        return self._observe(time_step), float(reward), terminated, truncated, {}


class _SeededDMControlVecEnv(TorchVectorEnvAdapter):
    """Defaults ``reset()`` to the configured ``seed`` when the caller omits
    one, so repeated evaluation rounds replay the same fixed episodes
    (mirrors ``rl_garden.envs.metaworld.env._SeededMetaWorldVecEnv``)."""

    def __init__(self, vec_env: gym.vector.VectorEnv, device: str, seed: int) -> None:
        super().__init__(vec_env, device)
        self._seed = seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return super().reset(seed=seed if seed is not None else self._seed, options=options)


def _make_env_fn(cfg: DMControlEnvConfig, seed: int):
    def _env_fn():
        # off_policy.py/base_algorithm.py read infos["final_info"]["episode"]
        # (same wrap as rl_garden.envs.mujoco.env).
        from gymnasium.wrappers import RecordEpisodeStatistics

        return RecordEpisodeStatistics(
            DMControlGymEnv(
                cfg.env_id,
                seed=seed,
                action_repeat=cfg.action_repeat,
                episode_frames=cfg.episode_frames,
                hard=cfg.hard,
                camera_id=cfg.camera_id,
                state=cfg.state,
                rgb_cameras=cfg.rgb_cameras,
                image_size=cfg.image_size,
            )
        )

    return _env_fn


def make_dm_control_env(cfg: DMControlEnvConfig):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

    vec_cls = AsyncVectorEnv if cfg.vectorization == "async" else SyncVectorEnv
    env_fns = [_make_env_fn(cfg, cfg.seed + i) for i in range(cfg.num_envs)]
    vec_env = vec_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    adapter: gym.vector.VectorEnv = _SeededDMControlVecEnv(vec_env, device=cfg.device, seed=cfg.seed)

    if cfg.frame_stack > 1:
        from rl_garden.envs.wrappers import ImageFrameStackWrapper

        image_keys = tuple(f"rgb_{cam}" for cam in cfg.rgb_cameras)
        adapter = ImageFrameStackWrapper(adapter, frame_stack=cfg.frame_stack, image_keys=image_keys)

    if cfg.reward_scale != 1.0 or cfg.reward_bias != 0.0:
        from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasVectorWrapper

        adapter = RewardScaleBiasVectorWrapper(adapter, scale=cfg.reward_scale, bias=cfg.reward_bias)

    return adapter
