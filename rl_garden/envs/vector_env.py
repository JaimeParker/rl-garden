"""Shared torch-native ``gymnasium.vector.VectorEnv`` adapter.

Wraps a numpy-based ``gymnasium.vector.VectorEnv`` (``SyncVectorEnv``/
``AsyncVectorEnv``) and exposes rl-garden's torch env contract:
observations/rewards/terminations/truncations as torch tensors, and
Gymnasium's own ``infos["final_obs"]``/``"_final_obs"`` (SAME_STEP
autoreset, ``dtype=object``, populated only at terminated/truncated
indices) translated into rl-garden's dense
``infos["final_observation"]``/``"_final_observation"`` convention --
matching ``ManiSkillVectorEnv.step()``'s exact key set
(``mani_skill/vector/wrappers/gymnasium.py:170-175``): ``final_observation``,
``final_info``, ``_final_info``, ``_final_observation``.

Replaces two backend-specific adapters that duplicated this translation with
a latent divergence between them: ``rl_garden.envs.minari.env
._TorchVectorEnvAdapter`` preserved ``final_info`` but filled non-terminated
``final_observation`` slots with zeros; ``rl_garden.envs.mujoco.env
._MujocoVecEnvAdapter`` filled non-terminated slots with the env's actual
current observation (matching ManiSkillVectorEnv) but silently dropped
``final_info``. This adapter combines the correct half of each: MuJoCo's
per-slot fill strategy for ``final_observation``, Minari's preservation of
``final_info``.

Note on ``VectorWrapper`` attribute assignment (verified against the
installed gymnasium version, not assumed): ``num_envs`` has no setter on
``VectorWrapper`` -- it's a read-only property forwarding to the wrapped
env -- so it is intentionally *not* assigned here.
``single_observation_space``/``single_action_space``/``action_space``/
``metadata`` all have setters and are assigned directly.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper


def _translate_box(space: gym.spaces.Box) -> gym.spaces.Box:
    if not np.issubdtype(space.dtype, np.floating):
        # uint8 image spaces etc. pass through unchanged (raw pixel values).
        return space
    return gym.spaces.Box(
        low=space.low.astype(np.float32),
        high=space.high.astype(np.float32),
        shape=space.shape,
        dtype=np.float32,
    )


def _translate_space(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({key: _translate_space(sub) for key, sub in space.spaces.items()})
    if isinstance(space, gym.spaces.Box):
        return _translate_box(space)
    return space


class TorchVectorEnvAdapter(VectorWrapper):
    """Wraps a numpy ``gymnasium.vector.VectorEnv`` to emit rl-garden's torch
    env contract. See module docstring for the ``final_observation``/
    ``final_info`` contract."""

    def __init__(self, vec_env: VectorEnv, device: str) -> None:
        super().__init__(vec_env)
        self.device = torch.device(device)
        self.single_observation_space = _translate_space(vec_env.single_observation_space)
        self.single_action_space = vec_env.single_action_space
        # off_policy.py's random-exploration phase reads the batched
        # action_space.shape directly (not single_action_space).
        self.action_space = vec_env.action_space
        self.metadata = dict(vec_env.metadata)
        self.metadata["autoreset_mode"] = AutoresetMode.SAME_STEP

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_tree(obs), info

    def step(self, actions: torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
        obs, reward, terminated, truncated, infos = self.env.step(actions_np)
        infos = self._translate_infos(infos, obs)
        return (
            self._convert_tree(obs),
            self._to_torch(reward),
            self._to_torch(terminated),
            self._to_torch(truncated),
            infos,
        )

    def _to_torch(self, array: Any) -> torch.Tensor:
        tensor = torch.as_tensor(array, device=self.device)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor

    def _convert_tree(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._convert_tree(v) for key, v in value.items()}
        return self._to_torch(value)

    def _translate_infos(self, infos: dict[str, Any], obs: Any) -> dict[str, Any]:
        out = dict(infos)
        final_obs_raw = out.pop("final_obs", None)
        mask = out.pop("_final_obs", None)
        if final_obs_raw is not None:
            out["final_observation"] = self._stack_final_obs(final_obs_raw, obs)
        if mask is not None:
            out["_final_observation"] = self._to_torch(mask)
        if "final_info" in out:
            out["final_info"] = self._convert_tree(out["final_info"])
        if "_final_info" in out:
            out["_final_info"] = self._to_torch(out["_final_info"])
        return out

    def _stack_final_obs(self, final_obs_raw: np.ndarray, obs: Any) -> Any:
        if isinstance(obs, dict):
            return {
                key: self._to_torch(
                    np.stack(
                        [
                            entry[key] if entry is not None else obs[key][i]
                            for i, entry in enumerate(final_obs_raw)
                        ]
                    )
                )
                for key in obs
            }
        return self._to_torch(
            np.stack(
                [entry if entry is not None else obs[i] for i, entry in enumerate(final_obs_raw)]
            )
        )
