"""CLI-level smoke test for the IQL off2on entrypoint."""

import json
from pathlib import Path

from _cli import run_entry


def test_iql_print_config_matches_paper_aligned_defaults(tmp_path):
    result = run_entry(
        "train_off2on",
        [
            "iql",
            "--print-config",
            "--log-type",
            "none",
            "--log-dir",
            str(tmp_path),
            "--env-backend",
            "robotwin",
        ],
    )
    assert result.returncode == 0, result.stderr

    config = json.loads(result.stdout)
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "iql"}
    assert config["inputs"]["warmup_steps"] == 0
    assert config["inputs"]["online_replay_mode"] == "mixed"
    assert config["inputs"]["offline_data_ratio"] == "auto"
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_iql_antmaze_paper_preset_resolves_key_values(tmp_path):
    result = run_entry(
        "train_off2on",
        [
            "iql",
            "--config",
            "configs/off2on/iql_antmaze_medium_play_v2_paper.yaml",
            "--print-config",
            "--log-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr

    config = json.loads(result.stdout)
    inputs = config["inputs"]
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "iql"}
    assert inputs["env_backend"] == "d4rl_legacy"
    assert inputs["dataset_backend"] == "d4rl_legacy"
    assert inputs["env_id"] == "antmaze-medium-play-v2"
    assert inputs["offline_dataset"] == "antmaze-medium-play-v2"
    assert inputs["obs"]["state"] is True
    assert inputs["obs"]["rgb"] == []
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
