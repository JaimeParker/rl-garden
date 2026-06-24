"""Small memory-mapped tensor store for disk-backed replay buffers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import torch
from gymnasium import spaces

MmapMode = Literal["create", "open"]
MMAP_FORMAT_VERSION = 1


def space_metadata(
    space: spaces.Space,
    *,
    dtype_resolver=None,
) -> dict[str, Any]:
    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                key: space_metadata(value, dtype_resolver=dtype_resolver)
                for key, value in space.spaces.items()
            },
        }
    if isinstance(space, spaces.Box):
        dtype = (
            dtype_resolver(space.dtype)
            if dtype_resolver is not None
            else torch.as_tensor(np.empty((), dtype=space.dtype)).dtype
        )
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(dtype).removeprefix("torch."),
        }
    raise TypeError(f"Unsupported replay-buffer space: {type(space).__name__}")


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    mapping = {
        torch.bool: np.dtype(np.bool_),
        torch.uint8: np.dtype(np.uint8),
        torch.int16: np.dtype(np.int16),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported memmap tensor dtype: {dtype}") from exc


class MmapTensorStore:
    """Owns a directory of memmap tensors and a strict schema manifest."""

    def __init__(
        self,
        directory: str | Path,
        *,
        mode: MmapMode,
        manifest: dict[str, Any],
    ) -> None:
        if mode not in ("create", "open"):
            raise ValueError(f"Unknown mmap mode: {mode!r}")

        self.directory = Path(directory)
        self.mode = mode
        self._maps: list[np.memmap] = []
        self._manifest_path = self.directory / "manifest.json"
        expected = {"format_version": MMAP_FORMAT_VERSION, **manifest}

        if mode == "create":
            if self.directory.exists() and any(self.directory.iterdir()):
                raise FileExistsError(
                    f"mmap directory {self.directory} already contains files; "
                    "use mmap_mode='open' to resume it"
                )
            self.directory.mkdir(parents=True, exist_ok=True)
            self._manifest_path.write_text(
                json.dumps(expected, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            if not self._manifest_path.is_file():
                raise FileNotFoundError(
                    f"mmap manifest not found: {self._manifest_path}"
                )
            actual = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(
                    "mmap manifest does not match the requested replay buffer"
                )

    def tensor(
        self,
        path: tuple[str, ...],
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        fill_value: int | float | bool = 0,
    ) -> torch.Tensor:
        if not path:
            raise ValueError("memmap tensor path must not be empty")
        encoded = [quote(component, safe="") for component in path]
        file_path = self.directory.joinpath(*encoded[:-1], f"{encoded[-1]}.bin")
        if self.mode == "create":
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.is_file():
            raise FileNotFoundError(f"memmap tensor file not found: {file_path}")
        array = np.memmap(
            file_path,
            dtype=_numpy_dtype(dtype),
            mode="w+" if self.mode == "create" else "r+",
            shape=shape,
        )
        if self.mode == "create" and fill_value != 0:
            array.fill(fill_value)
        self._maps.append(array)
        return torch.from_numpy(array)

    def flush(self) -> None:
        for array in self._maps:
            array.flush()
