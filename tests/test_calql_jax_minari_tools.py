from collections import OrderedDict
from io import BytesIO
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from tools.conversion.export_calql_minari_dataset import (
    array_sha256,
    flatten_buffer_observations,
)
from baselines.core.wire_protocol import read_status, write_error
from baselines.core.env_server import flatten_observation
from baselines.core.dataset import load_npz_dataset
from baselines.core.evaluation import summarize_episodes
from baselines.cal_ql.run_offline import (
    DATASET_KEYS,
    _load_dataset,
    flatten_legacy_observation,
)


def test_minari_observation_flattening_matches_combined_extractor_order():
    observation = {
        "achieved_goal": np.array([1.0, 2.0], dtype=np.float64),
        "desired_goal": np.array([3.0, 4.0], dtype=np.float64),
        "observation": np.array([5.0, 6.0, 7.0], dtype=np.float64),
    }

    flat = flatten_observation(
        observation, ("achieved_goal", "desired_goal", "observation")
    )

    assert flat.dtype == np.float32
    np.testing.assert_array_equal(flat, np.arange(1.0, 8.0, dtype=np.float32))


def test_exporter_flattens_buffer_using_observation_space_order():
    observation_space = spaces.Dict(
        OrderedDict(
            (
                ("achieved_goal", spaces.Box(-np.inf, np.inf, (2,), np.float32)),
                ("desired_goal", spaces.Box(-np.inf, np.inf, (2,), np.float32)),
                ("observation", spaces.Box(-np.inf, np.inf, (1,), np.float32)),
            )
        )
    )
    buffer = SimpleNamespace(
        obs={
            "achieved_goal": torch.tensor([[[1.0, 2.0]], [[6.0, 7.0]]]),
            "desired_goal": torch.tensor([[[3.0, 4.0]], [[8.0, 9.0]]]),
            "observation": torch.tensor([[[5.0]], [[10.0]]]),
        }
    )

    flat = flatten_buffer_observations(
        buffer, observation_space, torch.arange(2), "obs"
    )

    np.testing.assert_array_equal(
        flat,
        np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
            dtype=np.float32,
        ),
    )


def test_legacy_adapter_inserts_target_between_goal_and_ant_state():
    observation = np.arange(29, dtype=np.float32)

    flat = flatten_legacy_observation(observation, np.array([40.0, 41.0]))

    assert flat.shape == (31,)
    np.testing.assert_array_equal(flat[:6], [0.0, 1.0, 40.0, 41.0, 2.0, 3.0])


def test_episode_summary_reports_sparse_success_and_wilson_interval():
    result = summarize_episodes([0.0, 1.0, 0.0, 1.0], [1000, 200, 1000, 300])

    assert result["successes"] == 2
    assert result["success_rate"] == 0.5
    assert result["average_return"] == 0.5
    low, high = result["success_rate_wilson95"]
    assert low < 0.5 < high


def test_bridge_protocol_propagates_server_errors():
    stream = BytesIO()
    write_error(stream, "test failure")
    stream.seek(0)

    with pytest.raises(RuntimeError, match="test failure"):
        read_status(stream)


def test_array_hash_depends_on_dtype_and_contents():
    base = np.array([1.0, 2.0], dtype=np.float32)

    assert array_sha256(base) == array_sha256(base.copy())
    assert array_sha256(base) != array_sha256(base.astype(np.float64))


def test_jax_loader_rejects_array_that_does_not_match_manifest(tmp_path):
    arrays = {
        "observations": np.zeros((2, 31), dtype=np.float32),
        "actions": np.zeros((2, 8), dtype=np.float32),
        "next_observations": np.zeros((2, 31), dtype=np.float32),
        "rewards": np.array([-5.0, 5.0], dtype=np.float32),
        "dones": np.array([0.0, 1.0], dtype=np.float32),
        "mc_returns": np.array([-500.0, 5.0], dtype=np.float32),
    }
    path = tmp_path / "dataset.npz"
    np.savez(path, **arrays)
    manifest = {
        "observation_keys": ["achieved_goal", "desired_goal", "observation"],
        "arrays": {
            key: {
                "shape": list(arrays[key].shape),
                "dtype": str(arrays[key].dtype),
                "sha256": array_sha256(arrays[key]),
            }
            for key in DATASET_KEYS
        },
    }
    manifest["arrays"]["actions"]["sha256"] = "0" * 64
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="hash mismatch for actions"):
        _load_dataset(path)


def test_load_npz_dataset_skips_optional_validation_when_unset(tmp_path):
    """Regression guard for the generalization out of Cal-QL's _load_dataset:
    a baseline that doesn't share Cal-QL's AntMaze observation-key order or
    sparse {-5, 5} reward assumption can omit those checks entirely."""
    arrays = {
        "observations": np.zeros((2, 4), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "next_observations": np.zeros((2, 4), dtype=np.float32),
        "rewards": np.array([0.1, 0.2], dtype=np.float32),
        "dones": np.array([0.0, 1.0], dtype=np.float32),
        "mc_returns": np.array([0.1, 0.2], dtype=np.float32),
    }
    path = tmp_path / "dataset.npz"
    np.savez(path, **arrays)
    manifest = {
        "observation_keys": ["some", "other", "order"],
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
            }
            for key, value in arrays.items()
        },
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))

    dataset, loaded_manifest = load_npz_dataset(path, keys=DATASET_KEYS)

    np.testing.assert_array_equal(dataset["rewards"], arrays["rewards"])
    assert loaded_manifest["observation_keys"] == ["some", "other", "order"]
