"""CLI-level smoke test for the AWAC off2on entrypoint."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_awac_print_config_matches_corl_aligned_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"RLG_LOG_TYPE": "wandb", "MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "awac",
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
    assert config["training_phase"] == "off2on"
    assert config["algorithm"] == "awac"
    assert config["args"]["warmup_steps"] == 0
    assert config["args"]["online_replay_mode"] == "mixed"
    assert config["args"]["offline_data_ratio"] == "auto"
    assert config["args"]["awac_lambda"] == 1.0
    assert config["args"]["exp_adv_max"] == 100.0
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr
