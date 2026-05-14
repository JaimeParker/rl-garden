"""Checkpoint and replay-buffer serialization helpers.

Checkpoints are torch-native ``.pt`` dictionaries. Model state and replay
state are intentionally split because replay buffers can be much larger than
policy/optimizer state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.buffers.dict_buffer import DictArray

FORMAT_VERSION = 1

# Map legacy algorithm class names (from checkpoints saved before the
# CQL/CalQL rename) to their current canonical class. Keeps existing
# pretrained checkpoints loadable after the public API moved from
# ``OfflineCQL`` / ``OfflineCalQL`` to ``CQL`` / ``CalQL``.
_ALGORITHM_CLASS_ALIASES: dict[str, str] = {
    "OfflineCQL": "CQL",
    "OfflineCalQL": "CalQL",
}


def _canonical_algorithm_class(name: Any) -> Any:
    """Resolve legacy algorithm class names to their current canonical form."""
    if isinstance(name, str):
        return _ALGORITHM_CLASS_ALIASES.get(name, name)
    return name


def space_metadata(space: spaces.Space) -> dict[str, Any]:
    """Return stable metadata for compatibility checks."""
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": tuple(int(v) for v in space.shape),
            "dtype": str(space.dtype),
        }
    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {k: space_metadata(v) for k, v in space.spaces.items()},
        }
    return {"type": type(space).__name__}


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    algorithm_class: str,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    strict: bool,
) -> None:
    metadata = checkpoint.get("metadata", {})
    errors: list[str] = []

    if checkpoint.get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version mismatch: checkpoint has {checkpoint.get('format_version')}, "
            f"expected {FORMAT_VERSION}"
        )
    checkpoint_algorithm = _canonical_algorithm_class(metadata.get("algorithm_class"))
    if checkpoint_algorithm != algorithm_class:
        errors.append(
            f"algorithm mismatch: checkpoint has {metadata.get('algorithm_class')!r}, "
            f"current agent is {algorithm_class!r}"
        )

    current_obs = space_metadata(observation_space)
    current_action = space_metadata(action_space)
    if metadata.get("observation_space") != current_obs:
        errors.append("observation_space metadata does not match current env")
    if metadata.get("action_space") != current_action:
        errors.append("action_space metadata does not match current env")

    if errors and strict:
        raise ValueError("Incompatible checkpoint:\n- " + "\n- ".join(errors))


def checkpoint_dict(
    *,
    algorithm_class: str,
    global_step: int,
    global_update: int,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    hyperparameters: dict[str, Any],
    state: dict[str, Any],
    replay_buffer_path: str | None = None,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "metadata": {
            "algorithm_class": algorithm_class,
            "global_step": int(global_step),
            "global_update": int(global_update),
            "observation_space": space_metadata(observation_space),
            "action_space": space_metadata(action_space),
            "hyperparameters": hyperparameters,
            "replay_buffer_path": replay_buffer_path,
        },
        "state": state,
    }


def save_checkpoint_file(path: str | Path, checkpoint: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return out


def load_checkpoint_file(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def replay_buffer_path_for_checkpoint(path: str | Path) -> Path:
    ckpt_path = Path(path)
    stem = ckpt_path.stem
    if stem.startswith("checkpoint_"):
        suffix = stem.removeprefix("checkpoint_")
    elif stem == "final":
        suffix = "final"
    else:
        suffix = stem
    return ckpt_path.with_name(f"replay_buffer_{suffix}.pt")


def _tensor_tree_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if isinstance(tree, dict):
        return {k: _tensor_tree_to_device(v, device) for k, v in tree.items()}
    return tree


def _dict_array_state(array: DictArray) -> dict[str, Any]:
    return {k: _dict_array_state(v) if isinstance(v, DictArray) else v for k, v in array.data.items()}


def _load_dict_array_state(array: DictArray, state: dict[str, Any], device: torch.device) -> None:
    if set(array.data) != set(state):
        raise ValueError(
            "Replay buffer observation keys do not match: "
            f"checkpoint={sorted(state)}, current={sorted(array.data)}"
        )
    for key, value in state.items():
        if isinstance(array.data[key], DictArray):
            _load_dict_array_state(array.data[key], value, device)
        else:
            tensor = value.to(device)
            if array.data[key].shape != tensor.shape:
                raise ValueError(
                    f"Replay buffer tensor shape mismatch for {key!r}: "
                    f"checkpoint={tuple(tensor.shape)}, current={tuple(array.data[key].shape)}"
                )
            array.data[key] = tensor


def replay_buffer_state_dict(buffer: BaseReplayBuffer) -> dict[str, Any]:
    obs_state = _dict_array_state(buffer.obs) if isinstance(buffer.obs, DictArray) else buffer.obs
    next_obs_state = (
        _dict_array_state(buffer.next_obs)
        if isinstance(buffer.next_obs, DictArray)
        else buffer.next_obs
    )
    state: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "buffer_class": type(buffer).__name__,
        "num_envs": int(buffer.num_envs),
        "buffer_size": int(buffer.buffer_size),
        "per_env_buffer_size": int(buffer.per_env_buffer_size),
        "pos": int(buffer.pos),
        "full": bool(buffer.full),
        "storage_device": str(buffer.storage_device),
        "sample_device": str(buffer.sample_device),
        "obs": obs_state,
        "next_obs": next_obs_state,
        "actions": buffer.actions,
        "rewards": buffer.rewards,
        "dones": buffer.dones,
    }
    if hasattr(buffer, "base_actions"):
        state["base_actions"] = buffer.base_actions
    if hasattr(buffer, "next_base_actions"):
        state["next_base_actions"] = buffer.next_base_actions
    if hasattr(buffer, "gamma"):
        state["gamma"] = float(buffer.gamma)
    if hasattr(buffer, "_mc_table"):
        state["mc_table"] = buffer._mc_table
    return state


def load_replay_buffer_state_dict(
    buffer: BaseReplayBuffer,
    state: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    errors = []
    if state.get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version mismatch: checkpoint has {state.get('format_version')}, "
            f"expected {FORMAT_VERSION}"
        )
    if state.get("buffer_class") != type(buffer).__name__:
        errors.append(
            f"buffer_class mismatch: checkpoint has {state.get('buffer_class')!r}, "
            f"current buffer is {type(buffer).__name__!r}"
        )
    for key in ("num_envs", "buffer_size", "per_env_buffer_size"):
        if int(state.get(key, -1)) != int(getattr(buffer, key)):
            errors.append(
                f"{key} mismatch: checkpoint has {state.get(key)}, "
                f"current buffer has {getattr(buffer, key)}"
            )
    if errors and strict:
        raise ValueError("Incompatible replay buffer checkpoint:\n- " + "\n- ".join(errors))

    storage_device = buffer.storage_device
    buffer.pos = int(state["pos"])
    buffer.full = bool(state["full"])

    if isinstance(buffer.obs, DictArray):
        _load_dict_array_state(buffer.obs, state["obs"], storage_device)
        _load_dict_array_state(buffer.next_obs, state["next_obs"], storage_device)
    else:
        buffer.obs = state["obs"].to(storage_device)
        buffer.next_obs = state["next_obs"].to(storage_device)

    buffer.actions = state["actions"].to(storage_device)
    buffer.rewards = state["rewards"].to(storage_device)
    buffer.dones = state["dones"].to(storage_device)
    for key in ("base_actions", "next_base_actions"):
        if not hasattr(buffer, key):
            continue
        if key not in state:
            if strict:
                raise ValueError(f"Replay buffer checkpoint is missing {key!r}.")
            continue
        setattr(buffer, key, state[key].to(storage_device))
    buffer.sample_device = torch.device(buffer.sample_device)

    if hasattr(buffer, "gamma") and "gamma" in state:
        buffer.gamma = float(state["gamma"])
    if hasattr(buffer, "_mc_table"):
        mc_table = state.get("mc_table")
        buffer._mc_table = None if mc_table is None else mc_table.to(storage_device)


def save_replay_buffer_file(path: str | Path, buffer: BaseReplayBuffer) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(replay_buffer_state_dict(buffer), out)
    return out


def load_replay_buffer_file(
    path: str | Path,
    buffer: BaseReplayBuffer,
    *,
    strict: bool = True,
) -> None:
    state = torch.load(Path(path), map_location=buffer.storage_device)
    load_replay_buffer_state_dict(buffer, state, strict=strict)
