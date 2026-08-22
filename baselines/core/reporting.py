"""Result/checkpoint I/O helpers shared across baseline orchestrators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_scalar(value) -> float:
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return float(np.mean(array))


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path, value) -> None:
    with open(path, "a") as target:
        target.write(json.dumps(value, sort_keys=True) + "\n")


def save_pickle_checkpoint(path, payload: dict) -> None:
    import cloudpickle

    with open(path, "wb") as target:
        cloudpickle.dump(payload, target)


def load_pickle_checkpoint(path) -> Any:
    import cloudpickle

    with open(path, "rb") as source:
        return cloudpickle.load(source)
