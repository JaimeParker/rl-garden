"""CLI-level smoke test for the Cal-QL off2on entrypoint."""

import json
import os
import subprocess
import sys
from pathlib import Path

from rl_garden.common.effective_config import apply_strict_mapping, load_preset
from rl_garden.training.off2on.calql import CalQLOff2OnArgs


def test_calql_print_config_matches_paper_aligned_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "calql",
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
    assert config["selection"] == {"training_phase": "off2on", "algorithm": "calql"}
    assert config["inputs"]["warmup_steps"] == 0
    assert config["inputs"]["online_replay_mode"] == "mixed"
    assert config["inputs"]["offline_data_ratio"] == "auto"
    assert config["inputs"]["online_use_cql_loss"] is True
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_wsrl_print_config_unaffected_by_calql_defaults(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_off2on.py",
            "wsrl",
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
    assert config["selection"]["algorithm"] == "wsrl"
    assert config["inputs"]["warmup_steps"] == 5000
    assert config["inputs"]["online_replay_mode"] == "empty"
    assert config["inputs"]["offline_data_ratio"] == 0.0


def test_d4rl_expansion_presets_are_strict_and_complete():
    repo_root = Path(__file__).resolve().parents[1]
    paths = sorted((repo_root / "configs" / "off2on").glob("calql_*_v*_*.yaml"))

    assert len(paths) == 15
    configs = {}
    for path in paths:
        args = CalQLOff2OnArgs()
        apply_strict_mapping(args, load_preset(path).values)
        configs[args.env_id] = args

    assert configs["pen-binary-v0"].offline_dataset is None
    assert configs["pen-binary-v0"].num_online_steps == 300_000
    assert configs["door-binary-v0"].sparse_negative_reward == -5.0
    assert configs["kitchen-mixed-v0"].offline_data_ratio == 0.25
    assert configs["kitchen-mixed-v0"].cql_importance_sample is False
    assert configs["relocate-expert-v1"].hidden_dim == 256
    assert configs["relocate-expert-v1"].num_eval_episodes == 10
