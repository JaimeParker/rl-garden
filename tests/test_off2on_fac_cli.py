"""CLI-level smoke tests for the FAC off2on and offline entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_off2on_fac_print_config_matches_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "fac",
            "--print-config",
            "--log-type",
            "none",
            "--log-dir",
            str(tmp_path),
            "--env-backend",
            "robotwin",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(result.stdout)
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "fac"}
    inputs = config["inputs"]
    assert inputs["gamma"] == 0.995
    assert inputs["q_agg"] == "min"
    assert inputs["normalize_q_loss"] is True
    assert inputs["fac_alpha"] == 1.0
    assert inputs["fac_lambda"] == 1.0
    assert inputs["fac_threshold"] == "batch_adaptive"
    assert inputs["logp_method"] == "exact"
    assert inputs["logp_hutch_probes"] == 8
    assert inputs["weight_type"] == "linear"
    assert inputs["bc_lr"] == 3e-4
    assert inputs["bc_batch_size"] is None
    assert inputs["bc_pretrain_epochs"] == 250
    assert inputs["bc_pretrain_steps"] is None
    assert inputs["alpha"] == 10.0
    assert list(tmp_path.iterdir()) == []


def test_offline_fac_print_config_matches_defaults(tmp_path, capsys):
    """In-process pattern following tests/test_off2on_fino_cli.py -- FAC's
    offline dataset is only validated for truthiness by --print-config (see
    algorithm_registry.py's _validate_config), so a non-existent path is
    enough to exercise config materialization without loading real data.
    Asserts the printed JSON config's selection and key input defaults."""
    from rl_garden.training.offline import registry

    dataset_path = tmp_path / "fac.h5"

    registry.run_cli(
        [
            "fac",
            "--offline_dataset",
            str(dataset_path),
            "--device",
            "cpu",
            "--log_type",
            "none",
            "--log_dir",
            str(tmp_path / "runs"),
            "--print-config",
        ]
    )

    config = json.loads(capsys.readouterr().out)
    assert config["selection"] == {"training_phase": "offline", "algorithm": "fac"}
    inputs = config["inputs"]
    assert inputs["gamma"] == 0.995
    assert inputs["fac_alpha"] == 1.0
    assert inputs["fac_lambda"] == 1.0
    assert inputs["fac_threshold"] == "batch_adaptive"
