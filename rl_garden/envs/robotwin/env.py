"""Public rl-garden RoboTwin vector environment."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.common.observation_view import ObservationView
from rl_garden.common.utils import get_device
from rl_garden.envs.robotwin.config import ImageResizeBackend, RoboTwinEnvConfig
from rl_garden.envs.robotwin.executor import ThreadedRoboTwinExecutor


class RoboTwinEnv(gym.Env):
    """Vectorized RoboTwin environment with rl-garden's env contract."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RoboTwinEnvConfig,
        executor: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if cfg.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {cfg.num_envs}.")
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = get_device(cfg.device)
        bounded_action = cfg.control_mode in {"delta_joint_pos", "ee_delta_pose"}
        self.single_action_space = spaces.Box(
            low=-1.0 if bounded_action else -np.inf,
            high=1.0 if bounded_action else np.inf,
            shape=(cfg.action_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (cfg.num_envs, cfg.action_dim)),
            high=np.broadcast_to(self.single_action_space.high, (cfg.num_envs, cfg.action_dim)),
            dtype=np.float32,
        )
        h, w = cfg.image_size
        self.image_keys = ["rgb"]
        if cfg.include_wrist_cameras:
            self.image_keys.extend(("rgb_left_wrist", "rgb_right_wrist"))
        obs_spaces: dict[str, spaces.Space] = {
            key: spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
            for key in self.image_keys
        }
        obs_spaces["state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        agent_spaces: dict[str, spaces.Space] = dict(obs_spaces)
        agent_key_map: list[tuple[str, str]] = []
        if cfg.agent_image_size is not None:
            agent_height, agent_width = cfg.agent_image_size
            for key in self.image_keys:
                agent_spaces[key] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(agent_height, agent_width, 3),
                    dtype=np.uint8,
                )
            for key in self.image_keys:
                source_key = f"agent_{key}"
                obs_spaces[source_key] = agent_spaces[key]
                agent_key_map.append((key, source_key))
            agent_key_map.extend(
                (key, key) for key in agent_spaces if key not in self.image_keys
            )
        self.single_observation_space = spaces.Dict(obs_spaces)
        self.observation_space = self.single_observation_space
        self.agent_observation_view = ObservationView(
            spaces.Dict(agent_spaces),
            tuple(agent_key_map),
        )

        if cfg.assets_path is not None:
            os.environ["ASSETS_PATH"] = cfg.assets_path
        self._generator = torch.Generator()
        self._generator.manual_seed(cfg.seed)
        self._current_seed_index = 0
        self._success_seeds = self._load_success_seeds()
        self.reset_state_ids = self._next_reset_state_ids()

        task_args = self._task_args()
        self.executor = executor or ThreadedRoboTwinExecutor(
            cfg,
            task_args=task_args,
            env_seeds=self.reset_state_ids.tolist(),
        )
        self.prev_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        del options
        if seed is not None:
            self._generator.manual_seed(seed)
        self.reset_state_ids = self._next_reset_state_ids()
        raw_obs = self.executor.reset(env_seeds=self.reset_state_ids.tolist())
        self._sync_actual_reset_state_ids(raw_obs)
        self._reset_metrics()
        return self._tensor_obs(raw_obs), {
            "instructions": [o.get("instruction") for o in raw_obs],
            "env_seed": self.reset_state_ids.clone(),
        }

    def step(self, actions):
        actions_np = self._actions_to_numpy(actions)
        results = self.executor.step(actions_np)
        raw_obs = [res.obs for res in results]
        rewards = torch.as_tensor([res.reward for res in results], dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor([res.terminated for res in results], dtype=torch.bool, device=self.device)
        truncations = torch.as_tensor([res.truncated for res in results], dtype=torch.bool, device=self.device)
        infos = _list_of_dict_to_dict([res.info for res in results])
        self.elapsed_steps += 1
        max_steps = self.cfg.max_episode_steps
        if max_steps is not None:
            truncations = torch.logical_or(truncations, self.elapsed_steps >= max_steps)
        infos = self._record_metrics(rewards, infos)
        if self.cfg.ignore_terminations:
            if "success" in infos:
                infos["episode"]["success_at_end"] = infos["success"].clone()
            terminations[:] = False
        obs = self._tensor_obs(raw_obs)
        dones = torch.logical_or(terminations, truncations)
        if self.cfg.auto_reset and dones.any():
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.executor.close()

    def qpos_targets_to_ee_pose(self, actions):
        """Convert batched ACT qpos targets without changing simulator state."""

        input_is_tensor = isinstance(actions, torch.Tensor)
        input_device = actions.device if input_is_tensor else None
        if input_is_tensor:
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        transformed = self.executor.qpos_targets_to_ee_pose(actions_np)
        if input_is_tensor:
            return torch.as_tensor(
                transformed,
                dtype=torch.float32,
                device=input_device,
            )
        return transformed

    def chunk_step(self, actions):
        actions_np = self._actions_to_numpy(actions)
        if actions_np.ndim != 3:
            raise ValueError(f"chunk_step expects [num_envs, horizon, action_dim], got {actions_np.shape}.")
        obs = None
        rewards = []
        terms = []
        truncs = []
        infos = None
        for t in range(actions_np.shape[1]):
            obs, reward, term, trunc, infos = self.step(actions_np[:, t, :])
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)
        assert obs is not None and infos is not None
        return (
            obs,
            torch.stack(rewards, dim=1),
            torch.stack(terms, dim=1),
            torch.stack(truncs, dim=1),
            infos,
        )

    def _actions_to_numpy(self, actions) -> np.ndarray:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        if actions_np.ndim == 3:
            if actions_np.shape[1] != 1:
                raise ValueError("Use chunk_step for multi-step action chunks.")
            actions_np = actions_np[:, 0, :]
        return actions_np

    def _tensor_obs(self, raw_obs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        h, w = self.cfg.image_size
        zero_img = np.zeros((h, w, 3), dtype=np.uint8)
        for key in self.image_keys:
            imgs = []
            for obs in raw_obs:
                img = obs.get(key)
                imgs.append(
                    zero_img
                    if img is None
                    else _resize_image(
                        img,
                        self.cfg.image_size,
                        backend=self.cfg.image_resize_backend,
                    )
                )
            out[key] = torch.as_tensor(
                np.stack(imgs), dtype=torch.uint8, device=self.device
            )
            if self.cfg.agent_image_size is not None:
                agent_imgs = [
                    _resize_image(
                        image,
                        self.cfg.agent_image_size,
                        backend=self.cfg.image_resize_backend,
                    )
                    for image in imgs
                ]
                out[f"agent_{key}"] = torch.as_tensor(
                    np.stack(agent_imgs), dtype=torch.uint8, device=self.device
                )
        states = [np.asarray(obs["state"], dtype=np.float32) for obs in raw_obs]
        out["state"] = torch.as_tensor(
            np.stack(states), dtype=torch.float32, device=self.device
        )
        return out

    def _handle_auto_reset(self, dones: torch.Tensor, obs, infos):
        final_obs = {k: v.clone() for k, v in obs.items()}
        final_info = _clone_info(infos)
        env_idx = torch.arange(self.num_envs, device=self.device)[dones].tolist()
        self._update_reset_state_ids(env_idx)
        seeds = [int(self.reset_state_ids[idx].item()) for idx in env_idx]
        raw_obs = self.executor.reset(env_indices=env_idx, env_seeds=seeds)
        self._sync_actual_reset_state_ids(raw_obs, env_idx)
        reset_obs = self._tensor_obs(raw_obs)
        for key, value in obs.items():
            value[dones] = reset_obs[key][dones]
        self._reset_metrics(env_idx)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _record_metrics(self, rewards: torch.Tensor, infos: dict[str, Any]) -> dict[str, Any]:
        self.returns += rewards
        if "success" in infos:
            success = _to_tensor(infos["success"], dtype=torch.bool, device=self.device)
            infos["success"] = success
            self.success_once = torch.logical_or(self.success_once, success)
        episode_len = torch.clamp(self.elapsed_steps, min=1)
        infos["episode"] = {
            "return": self.returns.clone(),
            "episode_len": episode_len.clone(),
            "reward": self.returns / episode_len,
            "success_once": self.success_once.clone(),
        }
        return infos

    def _reset_metrics(self, env_idx: Optional[list[int]] = None) -> None:
        if env_idx is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
        self.prev_reward[mask] = 0
        self.success_once[mask] = False
        self.fail_once[mask] = False
        self.returns[mask] = 0
        self.elapsed_steps[mask] = 0

    def _task_args(self) -> dict[str, Any]:
        args = dict(self.cfg.task_config)
        args.setdefault("task_name", self.cfg.task_name)
        args.setdefault("step_lim", self.cfg.step_lim or self.cfg.max_episode_steps or 400)
        args.setdefault("planner_backend", self.cfg.planner_backend)
        args.setdefault("embodiment", self.cfg.embodiment)
        args.setdefault("clear_cache_freq", self.cfg.clear_cache_freq)
        return args

    def _load_success_seeds(self) -> Optional[torch.Tensor]:
        if self.cfg.seeds_path is None or not os.path.exists(self.cfg.seeds_path):
            return None
        with open(self.cfg.seeds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seeds = data.get(self.cfg.task_name, {}).get("success_seeds")
        if not seeds:
            return None
        shuffled = torch.as_tensor(seeds, dtype=torch.long)
        indices = torch.randperm(shuffled.numel(), generator=self._generator)
        return shuffled[indices]

    def _next_reset_state_ids(self) -> torch.Tensor:
        if self._success_seeds is not None:
            group_size = max(1, self.cfg.group_size)
            num_group = max(1, self.num_envs // group_size)
            idx = (torch.arange(num_group) + self._current_seed_index) % self._success_seeds.numel()
            seeds = self._success_seeds[idx].repeat_interleave(group_size)[: self.num_envs]
            self._current_seed_index = (self._current_seed_index + num_group) % self._success_seeds.numel()
            return seeds
        return torch.randint(10_000, 200_000, (self.num_envs,), generator=self._generator)

    def _update_reset_state_ids(self, env_idx: list[int]) -> None:
        if self.cfg.use_fixed_reset_state_ids:
            return
        new_ids = self._next_reset_state_ids()
        for idx in env_idx:
            self.reset_state_ids[idx] = new_ids[idx]

    def _sync_actual_reset_state_ids(
        self,
        raw_obs: list[dict[str, Any]],
        env_idx: Optional[list[int]] = None,
    ) -> None:
        indices = list(range(self.num_envs)) if env_idx is None else list(env_idx)
        for offset, idx in enumerate(indices):
            source_idx = idx if len(raw_obs) == self.num_envs else offset
            if source_idx >= len(raw_obs):
                continue
            actual_seed = raw_obs[source_idx].get("_env_seed")
            if actual_seed is not None:
                self.reset_state_ids[idx] = int(actual_seed)


def make_robotwin_env(cfg: RoboTwinEnvConfig) -> RoboTwinEnv:
    return RoboTwinEnv(cfg)


def _resize_image(
    image: Any,
    size: tuple[int, int],
    *,
    backend: ImageResizeBackend = "pillow",
) -> np.ndarray:
    image = np.asarray(image)
    h, w = size
    if image.shape[:2] == (h, w):
        return image.astype(np.uint8, copy=False)
    image = image.astype(np.uint8)
    if backend == "pillow":
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((w, h)))
    if backend == "opencv":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "RoboTwin OpenCV image resizing requires OpenCV. Install the "
                "'rl_garden[robotwin]' optional dependencies."
            ) from exc

        return cv2.resize(
            image,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
    raise ValueError(f"Unsupported RoboTwin image resize backend: {backend!r}.")


def _list_of_dict_to_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, list[Any]] = {}
    for item in items:
        for key, value in item.items():
            out.setdefault(key, []).append(value)
    return out


def _to_tensor(value, dtype, device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _clone_info(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = _clone_info(value)
        else:
            out[key] = value.copy() if hasattr(value, "copy") else value
    return out
