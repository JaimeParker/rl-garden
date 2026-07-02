"""Minari-backed vectorized env: N ``recover_environment()`` copies batched
with ``gymnasium.vector.SyncVectorEnv``, wrapped in a thin torch-tensor
adapter matching rl-garden's env-backend contract.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from rl_garden.envs.minari.config import MinariEnvConfig


def _to_tensor(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {key: _to_tensor(value, device) for key, value in x.items()}
    tensor = torch.as_tensor(x, device=device)
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


class _TorchVectorEnvAdapter:
    """Wraps a numpy ``gymnasium.vector.VectorEnv`` to emit torch tensors.

    ``SyncVectorEnv(..., autoreset_mode=SAME_STEP)`` already produces
    ``infos["final_info"]``/``infos["_final_info"]`` in the nested-dict shape
    rl-garden's rollout/eval code expects (matching ManiSkillVectorEnv's
    convention) -- only the leaf numpy arrays need torch-ifying. The one key
    that needs renaming is ``infos["final_obs"]`` (Gymnasium's name) ->
    ``infos["final_observation"]`` (what ``off_policy.py`` reads), and its
    masked-object-array shape needs stacking into a batched tensor.
    """

    def __init__(self, vec_env, device: str | torch.device) -> None:
        self._vec_env = vec_env
        self.device = torch.device(device)
        self.num_envs = vec_env.num_envs
        self.single_observation_space = vec_env.single_observation_space
        self.single_action_space = vec_env.single_action_space

    def reset(self, *, seed: Optional[int] = None):
        obs, info = self._vec_env.reset(seed=seed)
        return _to_tensor(obs, self.device), info

    def step(self, actions: torch.Tensor):
        actions_np = (
            actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        )
        obs, rewards, terminations, truncations, infos = self._vec_env.step(actions_np)
        return (
            _to_tensor(obs, self.device),
            torch.as_tensor(rewards, device=self.device, dtype=torch.float32),
            torch.as_tensor(terminations, device=self.device, dtype=torch.bool),
            torch.as_tensor(truncations, device=self.device, dtype=torch.bool),
            self._translate_infos(infos),
        )

    def _translate_infos(self, infos: dict) -> dict:
        out = dict(infos)
        if "final_obs" in infos:
            out["final_observation"] = self._stack_final_obs(
                infos["final_obs"], infos["_final_obs"]
            )
            del out["final_obs"]
            del out["_final_obs"]
        if "final_info" in infos:
            out["final_info"] = _to_tensor(infos["final_info"], self.device)
            out["_final_info"] = torch.as_tensor(
                infos["_final_info"], device=self.device, dtype=torch.bool
            )
        return out

    def _stack_final_obs(self, final_obs_arr: np.ndarray, mask: np.ndarray) -> Any:
        template_idx = int(np.flatnonzero(mask)[0])
        template = final_obs_arr[template_idx]
        if isinstance(template, dict):
            return {
                key: torch.as_tensor(
                    np.stack(
                        [
                            final_obs_arr[i][key] if mask[i] else np.zeros_like(template[key])
                            for i in range(self.num_envs)
                        ]
                    ),
                    device=self.device,
                )
                for key in template.keys()
            }
        stacked = np.stack(
            [
                final_obs_arr[i] if mask[i] else np.zeros_like(template)
                for i in range(self.num_envs)
            ]
        )
        return torch.as_tensor(stacked, device=self.device)

    def close(self) -> None:
        self._vec_env.close()


def make_minari_env(cfg: MinariEnvConfig) -> _TorchVectorEnvAdapter:
    import gymnasium.vector
    import minari
    from gymnasium.wrappers import RecordEpisodeStatistics

    dataset = minari.load_dataset(cfg.dataset_id, download=cfg.download)

    def _make_sub_env():
        return RecordEpisodeStatistics(dataset.recover_environment(eval_env=cfg.eval_env))

    env_fns = [_make_sub_env for _ in range(cfg.num_envs)]
    vec_env = gymnasium.vector.SyncVectorEnv(
        env_fns, autoreset_mode=gymnasium.vector.AutoresetMode.SAME_STEP
    )
    return _TorchVectorEnvAdapter(vec_env, device=cfg.device)
