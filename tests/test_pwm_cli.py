"""CLI tests for the offline ``pwm`` entrypoint: --print-config defaults,
--dry-run on a synthetic dataset + converted world model, preflight errors,
and a few real training steps through the bespoke runner."""
from __future__ import annotations

import json

import pytest
import torch
from test_pwm_smoke import write_converted_world_model, write_dataset


def _required(tmp_path, *, task="task_b"):
    dataset_dir = write_dataset(tmp_path / "data")
    world_model_dir = write_converted_world_model(tmp_path / "wm")
    return [
        "--dataset_dir",
        str(dataset_dir),
        "--mmap_dir",
        str(tmp_path / "mmap"),
        "--world_model_checkpoint",
        str(world_model_dir),
        "--task",
        task,
    ]


def test_pwm_print_config_defaults(capsys):
    from rl_garden.training.offline import registry

    registry.run_cli(
        [
            "pwm",
            "--dataset_dir", "/tmp/x",
            "--mmap_dir", "/tmp/y",
            "--world_model_checkpoint", "/tmp/z",
            "--task", "walker-walk",
            "--log_type", "none",
            "--print-config",
        ]
    )
    config = json.loads(capsys.readouterr().out)
    assert config["selection"] == {"training_phase": "offline", "algorithm": "pwm"}
    inputs = config["inputs"]
    assert inputs["task"] == "walker-walk"
    assert inputs["horizon"] == 16
    assert inputs["batch_size"] == 512
    assert inputs["actor_units"] == [400, 200, 100]
    assert inputs["critic_units"] == [400, 200]
    assert inputs["num_critics"] == 3
    assert inputs["critic_iterations"] == 8
    assert inputs["critic_batches"] == 4
    assert inputs["min_logstd"] == -1.427
    assert inputs["num_offline_steps"] == 10_000
    assert inputs["eval_freq"] == 200
    assert inputs["eval_runs"] == 10
    assert inputs["seed"] == 1


@pytest.mark.parametrize("missing", ["--task", "--world_model_checkpoint", "--dataset_dir", "--mmap_dir"])
def test_pwm_preflight_requires_flags(missing):
    from rl_garden.training.offline import registry

    argv = {
        "--dataset_dir": "/tmp/x",
        "--mmap_dir": "/tmp/y",
        "--world_model_checkpoint": "/tmp/z",
        "--task": "walker-walk",
    }
    del argv[missing]
    flat = [item for pair in argv.items() for item in pair]
    with pytest.raises(SystemExit, match=f"{missing} is required for pwm"):
        registry.run_cli(["pwm", *flat, "--log_type", "none", "--print-config"])


def test_pwm_dry_run_materializes_config(tmp_path, capsys):
    from rl_garden.training.offline import registry

    registry.run_cli(["pwm", *_required(tmp_path), "--device", "cpu", "--log_type", "none", "--dry-run"])
    output = capsys.readouterr().out
    assert "rl_garden.algorithms.pwm.PWM" in output
    assert not (tmp_path / "mmap").exists()  # dry-run uses a temporary mmap dir


def test_pwm_dry_run_rejects_unknown_task(tmp_path):
    from rl_garden.training.offline import registry

    with pytest.raises(SystemExit, match="not in the dataset's tasks"):
        registry.run_cli(
            ["pwm", *_required(tmp_path, task="nope"), "--device", "cpu", "--log_type", "none", "--dry-run"]
        )


def test_pwm_cli_trains_and_saves_checkpoint(tmp_path):
    from rl_garden.training.offline import registry

    checkpoint_dir = tmp_path / "ckpt"
    registry.run_cli(
        [
            "pwm",
            *_required(tmp_path),
            "--device", "cpu",
            "--horizon", "3",
            "--batch_size", "8",
            "--actor_units", "16", "16",
            "--critic_units", "16",
            "--critic_iterations", "1",
            "--critic_batches", "2",
            "--num_offline_steps", "3",
            "--eval_freq", "0",
            "--log_freq", "1",
            "--log_type", "none",
            "--log_dir", str(tmp_path / "runs"),
            "--checkpoint_dir", str(checkpoint_dir),
        ]
    )
    payload = torch.load(checkpoint_dir / "pwm_offline.pt", map_location="cpu", weights_only=False)
    assert payload["metadata"]["algorithm_class"] == "PWM"
    assert payload["metadata"]["global_update"] == 3
    assert payload["metadata"]["hyperparameters"]["task_idx"] == 1
