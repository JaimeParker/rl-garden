"""CLI-level tests for the DreamerV3 online entrypoint: --print-config for
state and rgb configs, contdisc/compute_dtype defaults, and asymmetric
obs_groups rejected at preflight (static, zero-instantiation -- DreamerV3
registers an ``algorithm_cls`` factory, see
``rl_garden/training/online/dreamer_v3.py``)."""
from __future__ import annotations

import json

import pytest


def test_dreamer_v3_print_config_state_defaults(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(
        [
            "dreamer_v3",
            "--log_type",
            "none",
            "--print-config",
        ]
    )

    config = json.loads(capsys.readouterr().out)
    assert config["selection"] == {"training_phase": "online", "algorithm": "dreamer_v3"}
    inputs = config["inputs"]
    assert inputs["size"] == "12M"
    assert inputs["contdisc"] is False
    assert inputs["compute_dtype"] is None
    assert inputs["batch_size"] == 16
    assert inputs["batch_length"] == 64
    assert inputs["train_ratio"] == 512.0
    assert inputs["imag_horizon"] == 15
    assert inputs["lam"] == 0.95
    assert inputs["lr"] == 4e-5
    assert inputs["warmup"] == 1_000
    assert inputs["slow_target_fraction"] == 0.02
    assert inputs["unimix"] == 0.01
    assert inputs["buffer_size"] == 1_000_000
    assert inputs["learning_starts"] == 1_024
    assert inputs["obs"]["rgb"] == []


def test_dreamer_v3_print_config_rgb_defaults(capsys):
    """--print-config is preflight-only (no env/agent construction -- see
    algorithm_registry.py's "print_config" vs "dry_run" command handling),
    so this only exercises CLI parsing + preflight for a visual obs config,
    not build_dreamer_v3's dreamer_conv-backbone override (that needs a
    constructed agent -- see the --dry-run test below, which needs a real
    sim env and therefore only runs on 6017)."""
    from rl_garden.training.online import registry

    registry.run_cli(
        [
            "dreamer_v3",
            "--log_type",
            "none",
            "--obs.rgb",
            "base_camera",
            "--obs.image-size",
            "64",
            "64",
            "--print-config",
        ]
    )

    config = json.loads(capsys.readouterr().out)
    assert config["selection"] == {"training_phase": "online", "algorithm": "dreamer_v3"}
    assert config["inputs"]["obs"]["rgb"] == ["base_camera"]
    assert config["inputs"]["obs"]["image_size"] == [64, 64]


def test_dreamer_v3_dry_run_rgb_sets_dreamer_conv_backbone():
    """--dry-run constructs the real env + agent (build_dreamer_v3's own
    dreamer_conv-backbone override only runs there) -- needs a real
    ManiSkill/Vulkan env, so this only runs on 6017 (mb-common.md rule 5)."""
    import io
    from contextlib import redirect_stdout

    from rl_garden.training.online import registry

    buf = io.StringIO()
    with redirect_stdout(buf):
        registry.run_cli(
            [
                "dreamer_v3",
                "--num_envs",
                "2",
                "--num_eval_envs",
                "2",
                "--buffer_size",
                "400",
                "--batch_size",
                "4",
                "--batch_length",
                "6",
                "--learning_starts",
                "40",
                "--log_type",
                "none",
                "--obs.rgb",
                "base_camera",
                "--obs.image-size",
                "64",
                "64",
                "--dry-run",
            ]
        )
    config = json.loads(buf.getvalue())
    encoder_config = config["algorithm"]["constructor_kwargs"]["encoder_config"]
    assert encoder_config["backbone"] == "dreamer_conv"


def test_dreamer_v3_asymmetric_obs_groups_rejected_at_preflight():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.observations import ObsGroups
    from rl_garden.observations.config import ObservationConfig
    from rl_garden.training.algorithm_registry import ParsedCommand
    from rl_garden.training.online import registry
    from rl_garden.training.online.dreamer_v3 import DreamerV3Args

    registry.discover()
    args = DreamerV3Args(
        obs=ObservationConfig(rgb=("base_camera",)),
        obs_groups=ObsGroups(actor=("state",), critic=("state", "rgb_base_camera")),
    )
    command = ParsedCommand(args, "dreamer_v3", "print_config", None, {}, (), {})

    with pytest.raises(ConfigError, match="only supports"):
        registry._validate_config(command)


def test_dreamer_v3_critic_encoder_rejected_at_preflight():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.encoders.config import EncoderConfig
    from rl_garden.training.algorithm_registry import ParsedCommand
    from rl_garden.training.online import registry
    from rl_garden.training.online.dreamer_v3 import DreamerV3Args

    registry.discover()
    args = DreamerV3Args(critic_encoder=EncoderConfig(features_dim=17))
    command = ParsedCommand(args, "dreamer_v3", "print_config", None, {}, (), {})

    with pytest.raises(ConfigError, match="only supports"):
        registry._validate_config(command)
