"""Shared torch-native ``gymnasium.vector.VectorEnv`` adapter.

Wraps a numpy-based ``gymnasium.vector.VectorEnv`` (``SyncVectorEnv``/
``AsyncVectorEnv``) and exposes rl-garden's torch env contract:
observations/rewards/terminations/truncations as torch tensors, and
Gymnasium's own ``infos["final_obs"]``/``"_final_obs"`` (SAME_STEP
autoreset, ``dtype=object``, populated only at terminated/truncated
indices) translated into the dense ``infos["final_observation"]``;
``final_info`` and ``_final_info`` are converted to tensors, and the optional
``decision``/``decision_components`` keys become dense ``torch.bool`` tensors
(their ``_decision*`` presence masks are dropped). See the episode boundary
contract in ``.agents/rules/adding-env-backend.md``.

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

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper
from gymnasium.vector.utils import batch_space

from rl_garden.common.spaces import canonicalize_floating_observation_space

# Optional per-step decision flags (episode boundary contract rules 9-10).
_DECISION_KEYS = ("decision", "decision_components")


def _row_present(values: Any, present: Any, row: int) -> bool:
    """Whether row ``row`` of a gymnasium-batched info entry was reported; a
    missing presence mask means every row was."""
    return values is not None and (present is None or bool(present[row]))


class TorchVectorEnvAdapter(VectorWrapper):
    """Wraps a numpy ``gymnasium.vector.VectorEnv`` to emit rl-garden's torch
    env contract. See the module docstring for the ``final_*`` translation."""

    def __init__(self, vec_env: VectorEnv, device: str) -> None:
        super().__init__(vec_env)
        self.device = torch.device(device)
        self.single_observation_space = canonicalize_floating_observation_space(
            vec_env.single_observation_space
        )
        self.observation_space = batch_space(self.single_observation_space, vec_env.num_envs)
        self.single_action_space = vec_env.single_action_space
        # off_policy.py's random-exploration phase reads the batched
        # action_space.shape directly (not single_action_space).
        self.action_space = vec_env.action_space
        self.metadata = dict(vec_env.metadata)
        self.metadata["autoreset_mode"] = AutoresetMode.SAME_STEP

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_tree(obs), info

    def step(self, actions):
        actions_np = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
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

    def _convert_info_tree(self, value: Any, *, field_mask: Any | None = None) -> Any:
        """Convert terminal info without losing Gymnasium's object-batched scalars.

        ``SyncVectorEnv`` represents some scalar info values as object arrays.
        OGBench scene tasks expose ``success`` this way, even though every
        element is a boolean.  Torch cannot consume an object array directly,
        so first restore its homogeneous NumPy dtype.  Object values that do
        not normalize to a tensor-compatible array remain an explicit error:
        terminal metrics must not be silently discarded.
        """
        if isinstance(value, dict):
            return {
                key: self._convert_info_tree(v, field_mask=value.get(f"_{key}"))
                for key, v in value.items()
            }
        if isinstance(value, np.ndarray) and value.dtype == object:
            entries = value.reshape(-1).tolist()
            missing = np.asarray([entry is None for entry in entries], dtype=bool)
            present = [entry for entry in entries if entry is not None]
            if missing.any():
                if field_mask is None:
                    raise TypeError(
                        "Cannot translate terminal info with missing object values without "
                        "its Gymnasium field mask."
                    )
                valid = np.asarray(field_mask, dtype=bool).reshape(-1)
                if valid.shape != missing.shape or np.any(missing & valid):
                    raise TypeError(
                        "Terminal info contains a missing value marked present by its "
                        "Gymnasium field mask."
                    )
                value = np.asarray(present)
                if value.dtype == object:
                    raise TypeError(
                        "Cannot translate an object-valued terminal info field to a tensor; "
                        "the field must contain homogeneous numeric or boolean values."
                    )
                filled = np.zeros(missing.shape, dtype=value.dtype)
                filled[~missing] = value
                value = filled.reshape(field_mask.shape)
            else:
                value = np.asarray(value.tolist())
            if value.dtype == object:
                raise TypeError(
                    "Cannot translate an object-valued terminal info field to a tensor; "
                    "the field must contain homogeneous numeric or boolean values."
                )
        return self._to_torch(value)

    def _translate_infos(self, infos: dict[str, Any], obs: Any) -> dict[str, Any]:
        out = dict(infos)
        final_obs_raw = out.pop("final_obs", None)
        out.pop("_final_obs", None)
        if final_obs_raw is not None:
            out["final_observation"] = self._stack_final_obs(final_obs_raw, obs)
        for key in _DECISION_KEYS:
            self._translate_decision_key(out, key)
        if "final_info" in out:
            out["final_info"] = self._convert_info_tree(out["final_info"])
        if "_final_info" in out:
            out["_final_info"] = self._to_torch(out["_final_info"])
        return out

    def _translate_decision_key(self, out: dict[str, Any], key: str) -> None:
        """Replace gymnasium's batched ``out[key]`` and its ``_key`` presence
        mask with one dense ``torch.bool`` tensor on the env device.

        Under SAME_STEP autoreset gymnasium moves the info of an env's last
        step into ``final_info`` and reports the reset info at the top level,
        so an ended row reads its flag from ``final_info`` (the flag belongs to
        the step that ended the episode) and every other row from the top
        level. Every row must report the key. Validation of dtype and shape is
        left to the consumer (episode boundary contract rules 9-10).
        """
        values = out.pop(key, None)
        present = out.pop(f"_{key}", None)
        final_info = out.get("final_info")
        final_values = final_info.get(key) if isinstance(final_info, dict) else None
        if values is None and final_values is None:
            return
        final_present = final_info.get(f"_{key}") if final_values is not None else None
        ended = out.get("_final_info")
        rows = []
        for i in range(self.num_envs):
            if ended is not None and ended[i] and _row_present(final_values, final_present, i):
                rows.append(np.asarray(final_values[i], dtype=bool))
            elif _row_present(values, present, i):
                rows.append(np.asarray(values[i], dtype=bool))
            else:
                raise ValueError(
                    f"Env {i} did not report infos[{key!r}] on this step; an env that "
                    "reports it must do so on every step (episode boundary contract)."
                )
        out[key] = torch.as_tensor(np.stack(rows), device=self.device)

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
