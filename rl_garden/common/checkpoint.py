"""Checkpoint and replay-buffer serialization helpers.

Checkpoints are torch-native ``.pt`` dictionaries. Model state and replay
state are intentionally split because replay buffers can be much larger than
policy/optimizer state.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Optional

import torch
import yaml
from gymnasium import spaces

from rl_garden.buffers.base import BaseReplayBuffer
from rl_garden.common.effective_config import _UNRECORDED_IMPORT_PATH_PREFIX
from rl_garden.common.spaces import canonicalize_floating_observation_space

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


def _observation_space_metadata(space: spaces.Space) -> dict[str, Any]:
    return space_metadata(canonicalize_floating_observation_space(space))


def _canonical_observation_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return metadata
    if metadata.get("type") == "Box" and str(metadata.get("dtype")).startswith("float"):
        return {**metadata, "dtype": "float32"}
    if metadata.get("type") == "Dict" and isinstance(metadata.get("spaces"), dict):
        return {
            **metadata,
            "spaces": {
                key: _canonical_observation_metadata(value)
                for key, value in metadata["spaces"].items()
            },
        }
    return metadata


#: What a checkpoint holds in place of a value that was or contained an absolute
#: path (``without_local_paths``). Not ``None``: the value was set.
REMOVED_PATH = "<absolute path removed>"

#: ``https://host/...``, ``s3://bucket/...``; not ``file:`` and not ``lmdb:///path``.
_URL = re.compile(r"\b(?!file:)[a-z][a-z0-9+.-]+://[^\s/\"'][^\s\"',;]*", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(
    r"""
      ^\s*/                                                  # POSIX
    | ^\s*~(?:$|[/A-Za-z_])
    | [\s"'`=:,;|@(\[{](?:/(?!\s)|~(?:/|[A-Za-z_][\w.-]*/))      # as a part of the text
    | \bfile:[/\\~]
    | [a-z][a-z0-9+.-]+:///
    | (?<![a-z0-9])[a-z]:[\\/](?!/)                          # Windows drive
    | \\\\[\w.$-]+\\[\w$]                                    # Windows share
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _text_without_paths(text: str) -> str:
    if _ABSOLUTE_PATH.search(_URL.sub("", text)) is None:
        return text
    if text.startswith(_UNRECORDED_IMPORT_PATH_PREFIX):
        return f"{_UNRECORDED_IMPORT_PATH_PREFIX}{REMOVED_PATH}>"
    if "\n" in text and not text.lstrip().startswith("{"):
        # A policy graph: only the node parameter that is a path goes.
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            document = None
        if isinstance(document, Mapping):
            return yaml.safe_dump(
                without_local_paths(document), sort_keys=True, default_flow_style=False
            )
    return REMOVED_PATH


def without_local_paths(value: Any) -> Any:
    """``value`` with each string (or bytes) that is or contains an absolute
    path (also ``~``, ``file:`` and Windows ones) replaced by ``REMOVED_PATH``,
    recursively through mappings (keys too), sequences and dataclass instances,
    and without the command line (``argv``). Containers are copied only where
    something changes, tensors never. The keys of a module state dict (parameter
    names, which loading matches exactly) are kept."""
    if isinstance(value, os.PathLike):
        return without_local_paths(os.fspath(value))
    if isinstance(value, str):
        return _text_without_paths(value)
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
        return value if _text_without_paths(text) == text else REMOVED_PATH.encode()
    if isinstance(value, Mapping):
        return _mapping_without_paths(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        cleaned = [without_local_paths(item) for item in value]
        if all(new is old for new, old in zip(cleaned, value)):
            return value
        return type(value)._make(cleaned) if hasattr(value, "_make") else type(value)(cleaned)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changed = {
            field.name: cleaned
            for field in dataclasses.fields(value)
            if (cleaned := without_local_paths(getattr(value, field.name)))
            is not getattr(value, field.name)
        }
        if not changed:
            return value
        value = copy.copy(value)
        for name, cleaned in changed.items():
            object.__setattr__(value, name, cleaned)
    return value


def _mapping_without_paths(value: Mapping) -> Mapping:
    items = {
        key: without_local_paths(item)
        for key, item in value.items()
        if not (key == "argv" and isinstance(item, (str, list, tuple)))
    }
    if not hasattr(value, "_metadata"):
        renamed = [key for key in items if without_local_paths(key) is not key]
        taken = set(items)
        for old in renamed:
            index = 0
            while f"{REMOVED_PATH} {index}" in taken:
                index += 1
            taken.add(f"{REMOVED_PATH} {index}")
            items = {
                f"{REMOVED_PATH} {index}" if key is old else key: item
                for key, item in items.items()
            }
    if items.keys() == value.keys() and all(items[key] is value[key] for key in items):
        return value
    if not isinstance(value, dict):
        return items
    # Keeps the type and the attributes.
    changed = copy.copy(value)
    changed.clear()
    changed.update(items)
    return changed


def holds_removed_path(value: Any) -> bool:
    """Whether ``without_local_paths`` took something out of ``value`` (also a
    part of a policy graph's text)."""
    if isinstance(value, (str, bytes)):
        marker = REMOVED_PATH if isinstance(value, str) else REMOVED_PATH.encode()
        return marker in value
    if isinstance(value, Mapping):
        return any(holds_removed_path(item) for pair in value.items() for item in pair)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(holds_removed_path(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(holds_removed_path(getattr(value, f.name)) for f in dataclasses.fields(value))
    return False


def _first_differing_line(checkpoint_text: str, current_text: str) -> str:
    checkpoint_lines = checkpoint_text.splitlines()
    current_lines = current_text.splitlines()
    for number, (stored, current) in enumerate(zip(checkpoint_lines, current_lines), start=1):
        if stored != current:
            return f"first difference at line {number}: checkpoint {stored!r}, agent {current!r}"
    number = min(len(checkpoint_lines), len(current_lines)) + 1
    return (
        f"first difference at line {number}: checkpoint has {len(checkpoint_lines)} lines, "
        f"agent has {len(current_lines)}"
    )


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    algorithm_class: str,
    compatible_algorithms: tuple[str, ...] = (),
    observation_space: spaces.Space,
    action_space: spaces.Space,
    strict: bool,
    policy_graph: Optional[str] = None,
) -> None:
    """``policy_graph`` is the current agent's canonical policy-graph text
    (``None`` without a graph); the checkpoint's must be present exactly
    when it is, and then equal to it (a graph differing only in ``detach``
    or an activation has the same parameter shapes)."""
    metadata = checkpoint.get("metadata", {})
    errors: list[str] = []

    if checkpoint.get("format_version") != FORMAT_VERSION:
        errors.append(
            f"format_version mismatch: checkpoint has {checkpoint.get('format_version')}, "
            f"expected {FORMAT_VERSION}"
        )
    checkpoint_algorithm = _canonical_algorithm_class(metadata.get("algorithm_class"))
    if checkpoint_algorithm not in compatible_algorithms:
        errors.append(
            f"algorithm mismatch: checkpoint has {metadata.get('algorithm_class')!r}, "
            f"current agent is {algorithm_class!r}"
        )

    current_obs = _observation_space_metadata(observation_space)
    current_action = space_metadata(action_space)
    if _canonical_observation_metadata(metadata.get("observation_space")) != current_obs:
        errors.append("observation_space metadata does not match current env")
    if metadata.get("action_space") != current_action:
        errors.append("action_space metadata does not match current env")
    hyperparameters = metadata.get("hyperparameters")
    # The checkpoint's graph has no absolute paths, but one saved earlier may.
    checkpoint_graph, policy_graph = without_local_paths(
        (
            hyperparameters.get("policy_graph") if isinstance(hyperparameters, dict) else None,
            policy_graph,
        )
    )
    if checkpoint_graph is not None and policy_graph is None:
        errors.append(
            "policy_graph mismatch: the checkpoint's policy was built from a policy "
            "graph but the current agent has none; pass the same --policy-graph "
            "(its canonical text is in the checkpoint's hyperparameters['policy_graph'])"
        )
    elif checkpoint_graph is None and policy_graph is not None:
        errors.append(
            "policy_graph mismatch: the current agent's policy is built from a policy "
            "graph but the checkpoint's is not; drop --policy-graph to load it"
        )
    elif checkpoint_graph is not None and checkpoint_graph != policy_graph:
        errors.append(
            "policy_graph mismatch: the checkpoint's policy graph differs from the "
            f"current agent's; {_first_differing_line(checkpoint_graph, policy_graph)}"
        )

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
            "observation_space": _observation_space_metadata(observation_space),
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


def load_filtered_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    prefix: str = "",
    strict: bool = False,
) -> None:
    """Load only the entries of ``state_dict`` under ``prefix`` into ``module``.

    ``module`` should be the submodule ``prefix`` names, not its parent --
    e.g. to load ``some_state_dict["actor.*"]`` into ``target.actor``, call
    ``load_filtered_state_dict(target.actor, some_state_dict, prefix="actor")``,
    not ``load_filtered_state_dict(target, ...)`` (the latter would strip
    the prefix and then fail to match ``target``'s own, still-prefixed,
    top-level keys). ``prefix=""`` loads ``state_dict`` into ``module``
    as-is, no stripping -- for when ``module`` already matches
    ``state_dict``'s own key namespace.

    Generic cross-algorithm primitive: any algorithm's own
    ``state_dict()["policy"]`` (or any other nested ``nn.Module`` state
    dict) can be filtered down to one submodule's weights and loaded into a
    freshly constructed module of matching shape -- without requiring the
    source and target to share a full env/obs-space binding the way
    ``BaseAlgorithm.load()`` does.
    """
    if not prefix:
        module.load_state_dict(state_dict, strict=strict)
        return
    key_prefix = f"{prefix}."
    filtered = {
        key[len(key_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(key_prefix)
    }
    if not filtered:
        available = sorted({key.split(".", 1)[0] for key in state_dict})
        raise ValueError(
            f"No state_dict keys found under prefix {prefix!r} "
            f"(available top-level prefixes: {available})."
        )
    module.load_state_dict(filtered, strict=strict)


def replay_buffer_path_for_checkpoint(path: str | Path) -> Path:
    return sidecar_path_for_checkpoint(path, "replay_buffer")


def sidecar_path_for_checkpoint(path: str | Path, prefix: str) -> Path:
    """``<prefix>_<suffix>.pt`` beside the checkpoint: the suffix of
    ``checkpoint_<suffix>.pt``, else the checkpoint's stem."""
    ckpt_path = Path(path)
    stem = ckpt_path.stem
    suffix = stem.removeprefix("checkpoint_") if stem.startswith("checkpoint_") else stem
    return ckpt_path.with_name(f"{prefix}_{suffix}.pt")


class _Unrecorded:
    def __repr__(self) -> str:
        return "UNRECORDED"


#: What ``recorded_file`` returns for a checkpoint without the slot: it
#: predates the slot. ``None`` in the slot means that no file was written.
UNRECORDED: Final = _Unrecorded()


def recorded_file(container: Mapping[str, Any], key_path: tuple[str, ...]) -> Any:
    """The file name at ``key_path`` in ``container``: a name, ``None`` (no
    file) or ``UNRECORDED`` (no such slot)."""
    value: Any = container
    for key in key_path:
        if not isinstance(value, Mapping) or key not in value:
            return UNRECORDED
        value = value[key]
    return value


def record_file(container: dict[str, Any], key_path: tuple[str, ...], name: str) -> None:
    """Name ``name`` at ``key_path``; every mapping above the leaf must exist."""
    parent: Any = container
    for depth, key in enumerate(key_path[:-1]):
        parent = parent.get(key) if isinstance(parent, Mapping) else None
        if not isinstance(parent, dict):
            raise KeyError(
                f"Cannot record {name} at {'.'.join(key_path)}: the checkpoint state has no "
                f"mapping at {'.'.join(key_path[: depth + 1])}."
            )
    parent[key_path[-1]] = name


def referenced_file(
    checkpoint_path: str | Path,
    reference: Any,
    *,
    unrecorded: Optional[Path],
    description: str,
) -> Optional[Path]:
    """The file a checkpoint's ``reference`` (``recorded_file``) points at, or
    ``None`` when there is none to read. A name is relative to the checkpoint
    and must exist; ``None`` means no file; ``UNRECORDED`` falls back to
    ``unrecorded`` when that file exists."""
    if reference is UNRECORDED:
        return unrecorded if unrecorded is not None and unrecorded.exists() else None
    if reference is None:
        return None
    path = Path(checkpoint_path).parent / reference
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint references {description} {path}, but it does not exist."
        )
    return path


@dataclasses.dataclass(frozen=True)
class SidecarBuffer:
    """A buffer ``BaseAlgorithm.save(include_replay_buffer=True)`` writes beside
    the checkpoint, as ``<prefix>_<suffix>.pt``, and
    ``load(load_replay_buffer=True)`` restores."""

    #: The file name's prefix; with underscores as spaces, also the buffer's
    #: name in messages.
    prefix: str
    #: The slot under ``state["extra"]`` that names the file (``None`` there:
    #: no file was written).
    reference_key: tuple[str, ...]
    #: What ``save`` writes; ``None``: nothing.
    buffer: Optional[BaseReplayBuffer]
    #: Called by ``load`` with the file and ``strict``.
    restore: Callable[[Path, bool], None]
    #: Without the slot, read the ``<prefix>_...`` file named like the
    #: checkpoint's replay file, when there is one.
    unrecorded_beside_replay: bool = False


def _tensor_tree_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if isinstance(tree, dict):
        return {k: _tensor_tree_to_device(v, device) for k, v in tree.items()}
    return tree


def replay_buffer_state_dict(buffer: BaseReplayBuffer) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "buffer_class": type(buffer).__name__,
        "num_envs": int(buffer.num_envs),
        "buffer_size": int(buffer.buffer_size),
        "per_env_buffer_size": int(buffer.per_env_buffer_size),
        "storage_device": str(buffer.storage_device),
        "sample_device": str(buffer.sample_device),
        **buffer.state_dict(),
    }


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
    buffer.load_state_dict(_tensor_tree_to_device(state, buffer.storage_device), strict=strict)


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
