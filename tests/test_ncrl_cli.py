"""CLI tests for the online ``ncrl`` entrypoint: ``--print-config`` defaults
and ablation flags, required-flag preflight, and ``--dry-run`` on the real
dm_control backend (``walker-walk``) with a synthetic pretrained checkpoint
and retrieval DB."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from _ncrl_online_synthetic import make_pretrain_artifacts


def _cli(*args: str) -> dict:
    from rl_garden.training.online import registry

    buf = io.StringIO()
    with redirect_stdout(buf):
        registry.run_cli(["ncrl", "--log_type", "none", *args])
    return json.loads(buf.getvalue())


def test_print_config_defaults():
    config = _cli("--pretrained_world_model", "/nonexistent/rssm.pt", "--retrieval_db", "/nonexistent/db.pt", "--print-config")
    assert config["selection"] == {"training_phase": "online", "algorithm": "ncrl"}
    inputs = config["inputs"]
    assert inputs["env_backend"] == "dm_control" and inputs["env_id"] == "walker-walk"
    assert inputs["num_envs"] == 1 and inputs["num_eval_envs"] == 5
    assert inputs["obs"]["state"] is False and inputs["obs"]["rgb"] == ["default"]
    assert inputs["obs"]["image_size"] == [64, 64]
    assert inputs["online_ratio"] == 0.75 and inputs["num_retrieve_episodes"] == 500
    assert inputs["use_guidance"] is True
    assert inputs["bc_train_steps"] == 150_000 and inputs["bc_batch_size"] == 256
    assert inputs["guidance_schedule"] == ["linear", 1.0, 0.1, 100_000]
    assert inputs["warmup_frames"] == 15_000
    assert (inputs["batch_size"], inputs["batch_length"], inputs["train_ratio"]) == (16, 64, 256.0)
    assert (inputs["imag_horizon"], inputs["discount"], inputs["lam"], inputs["actor_ent"]) == (16, 0.99, 0.95, 1e-4)
    assert (inputs["actor_lr"], inputs["critic_lr"], inputs["model_lr"]) == (8e-5, 8e-5, 1e-4)
    assert inputs["kl_free_mode"] == "mean" and inputs["backbone"] == "ncrl"
    assert inputs["learning_starts"] is None and inputs["padded_action_dim"] is None


def test_ablation_flags_parse():
    config = _cli(
        "--pretrained_world_model", "/nonexistent/rssm.pt",
        "--online_ratio", "1.0",
        "--no-use_guidance",
        "--print-config",
    )
    assert config["inputs"]["online_ratio"] == 1.0
    assert config["inputs"]["use_guidance"] is False
    assert config["inputs"]["retrieval_db"] is None


def _validate(**fields):
    from rl_garden.training.algorithm_registry import ParsedCommand
    from rl_garden.training.online import registry
    from rl_garden.training.online.ncrl import NCRLArgs

    registry.discover()
    command = ParsedCommand(NCRLArgs(**fields), "ncrl", "print_config", None, {}, (), {})
    registry._validate_config(command)


def test_missing_pretrained_world_model_rejected():
    from rl_garden.common.effective_config import ConfigError

    with pytest.raises(ConfigError, match="--pretrained_world_model is required for ncrl"):
        _validate(retrieval_db="/x/db.pt")
    with pytest.raises(SystemExit, match="--pretrained_world_model is required"):
        _cli("--retrieval_db", "/x/db.pt", "--print-config")


def test_rehearsal_without_retrieval_db_rejected():
    from rl_garden.common.effective_config import ConfigError

    with pytest.raises(ConfigError, match="--retrieval_db is required"):
        _validate(pretrained_world_model="/x/rssm.pt")
    _validate(pretrained_world_model="/x/rssm.pt", online_ratio=1.0)


def test_dry_run_on_dm_control_constructs_agent(tmp_path):
    pytest.importorskip("dm_control")
    artifacts = make_pretrain_artifacts(tmp_path)
    config = _cli(
        "--pretrained_world_model", str(artifacts["checkpoint"]),
        "--retrieval_db", str(artifacts["retrieval_db"]),
        "--backbone", "dreamer_v3",  # the synthetic checkpoint's backbone
        "--num_eval_envs", "1",
        "--buffer_size", "2000",
        "--buffer_device", "cpu",
        "--compute_dtype", "float32",
        "--dry-run",
    )
    algorithm = config["algorithm"]
    assert algorithm["target"].endswith("NCRL")
    kwargs = algorithm["constructor_kwargs"]
    assert kwargs["pretrained_world_model"] == str(artifacts["checkpoint"])
    assert kwargs["episode_frames"] == 1000 and kwargs["action_repeat"] == 2
    assert kwargs["guidance_schedule"] == ["linear", 1.0, 0.1, 100_000]
    assert config["runtime"]["train_action_space"]["shape"] == [6]  # walker: real action dim
