"""H5-parsing helpers shared by H5-backed offline dataset loaders.

Format-agnostic within HDF5: knows how to open a file and read a node into
plain Python/NumPy structures, nothing about trajectory semantics.
"""

from __future__ import annotations

from typing import Any


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without h5py.
        raise ImportError(
            "Loading H5 datasets requires h5py. Install rl-garden with "
            "the ManiSkill/offline dependencies or install h5py directly."
        ) from exc
    return h5py


def _read_node(node: Any) -> Any:
    h5py = _require_h5py()
    if isinstance(node, h5py.Dataset):
        return node[()]
    if isinstance(node, h5py.Group):
        return {key: _read_node(node[key]) for key in node.keys()}
    raise TypeError(f"Unsupported H5 node type: {type(node)!r}")
