"""Loading and provenance helpers for the Minari-derived ``.npz`` datasets
baseline orchestrators train on, shared across baseline orchestrators.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

DEFAULT_DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "dones",
    "mc_returns",
)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def source_metadata(source, tracked_files):
    """Commit + per-file sha256 for an official baseline checkout, for
    provenance capture in a run's ``config.json``."""
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "path": str(source),
        "commit": commit,
        "sha256": {name: sha256_file(Path(source) / name) for name in tracked_files},
    }


def load_npz_dataset(
    path,
    *,
    keys=DEFAULT_DATASET_KEYS,
    expected_observation_keys=None,
    expected_reward_values=None,
):
    """Load a dataset ``.npz`` + its ``.manifest.json`` sidecar, verifying
    array shapes/dtypes/sha256 against the manifest.

    ``expected_observation_keys``/``expected_reward_values`` are optional
    extra validations (Cal-QL's AntMaze export checks both; a baseline
    without those assumptions can omit them).
    """
    manifest_path = Path(path).with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("dataset manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    if (
        expected_observation_keys is not None
        and manifest.get("observation_keys") != list(expected_observation_keys)
    ):
        raise ValueError("dataset manifest has an unexpected observation key order")

    with np.load(path) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise ValueError("dataset archive is missing keys: {}".format(missing))
        dataset = {key: np.asarray(archive[key], dtype=np.float32) for key in keys}
    count = dataset["rewards"].shape[0]
    if any(value.shape[0] != count for value in dataset.values()):
        raise ValueError("dataset arrays have inconsistent transition counts")
    if dataset["observations"].shape != dataset["next_observations"].shape:
        raise ValueError("observation and next-observation shapes differ")
    for key, value in dataset.items():
        expected = manifest["arrays"][key]
        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
            raise ValueError("dataset array metadata mismatch for {}".format(key))
        if sha256_array(value) != expected["sha256"]:
            raise ValueError("dataset array hash mismatch for {}".format(key))
    if expected_reward_values is not None and set(
        np.unique(dataset["rewards"]).tolist()
    ) != set(expected_reward_values):
        raise ValueError(
            "expected transformed rewards {}".format(expected_reward_values)
        )
    return dataset, manifest
