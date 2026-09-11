"""CLI-level smoke tests for the FloQ off2on and offline entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_off2on_floq_print_config_matches_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "floq",
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
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "floq"}
    inputs = config["inputs"]
    assert inputs["r_min"] == -1.0
    assert inputs["flow_num_ensembles"] == 2
    assert inputs["critic_flow_steps"] == 8
    assert inputs["alpha"] == 10.0
    assert inputs["critic_flow_net_arch"] is None
    assert list(tmp_path.iterdir()) == []


def test_offline_floq_print_config_matches_defaults(tmp_path, capsys):
    """In-process pattern following tests/test_diffusion_bc_cli.py -- FloQ's
    offline dataset is only validated for truthiness by --print-config (see
    algorithm_registry.py's _validate_config), so a non-existent path is
    enough to exercise config materialization without loading real data.
    Asserts the printed JSON config's selection and key input defaults."""
    from rl_garden.training.offline import registry

    dataset_path = tmp_path / "floq.h5"

    registry.run_cli(
        [
            "floq",
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
    assert config["selection"] == {"training_phase": "offline", "algorithm": "floq"}
    inputs = config["inputs"]
    assert inputs["r_min"] == -1.0
    assert inputs["critic_flow_steps"] == 8
    assert inputs["alpha"] == 10.0
