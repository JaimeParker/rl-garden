"""CLI-level smoke test for the AWAC off2on entrypoint."""

import json

from _cli import run_entry


def test_awac_print_config_matches_corl_aligned_defaults(tmp_path):
    result = run_entry(
        "train_off2on",
        [
            "awac",
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
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "awac"}
    assert config["inputs"]["warmup_steps"] == 0
    assert config["inputs"]["online_replay_mode"] == "mixed"
    assert config["inputs"]["offline_data_ratio"] == "auto"
    assert config["inputs"]["awac_lambda"] == 1.0
    assert config["inputs"]["exp_adv_max"] == 100.0
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr
