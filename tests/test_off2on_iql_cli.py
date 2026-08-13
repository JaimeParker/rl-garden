"""CLI-level smoke test for the IQL off2on entrypoint."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_iql_print_config_matches_paper_aligned_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "iql",
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
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "iql"}
    assert config["inputs"]["warmup_steps"] == 0
    assert config["inputs"]["online_replay_mode"] == "mixed"
    assert config["inputs"]["offline_data_ratio"] == "auto"
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_iql_antmaze_paper_preset_resolves_key_values(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "iql",
            "--config",
            "configs/off2on/iql_antmaze_medium_play_v2_paper.yaml",
            "--print-config",
            "--log-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(result.stdout)
    inputs = config["inputs"]
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "iql"}
    assert inputs["env_backend"] == "d4rl_legacy"
    assert inputs["dataset_backend"] == "d4rl_legacy"
    assert inputs["env_id"] == "antmaze-medium-play-v2"
    assert inputs["offline_dataset"] == "antmaze-medium-play-v2"
    assert inputs["obs_mode"] == "state"
    assert inputs["seed"] == 0
    assert inputs["actor_lr"] == 0.0001
    assert inputs["critic_value_lr"] == 0.0003
    assert inputs["offline_data_ratio"] == 0.5
    assert inputs["num_eval_episodes"] == 100
    assert inputs["num_eval_steps"] == 100000
    assert inputs["bootstrap_at_done"] == "truncated"
    assert inputs["actor_use_layer_norm"] is False
    assert inputs["critic_use_layer_norm"] is False
    assert inputs["value_use_layer_norm"] is False
    assert list(tmp_path.iterdir()) == []
