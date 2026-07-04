from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class _RegistryA(BaseAlgorithmRegistry):
    package_name = "unused.a"
    phase_name = "a"


class _RegistryB(BaseAlgorithmRegistry):
    package_name = "unused.b"
    phase_name = "b"


@dataclass
class _Args:
    value: int = 1


def test_registry_instances_are_isolated_and_entries_are_copied():
    first = _RegistryA()
    second = _RegistryB()
    first.register("only", _Args, lambda args: None)

    entries = first.entries()
    entries.clear()

    assert set(first.entries()) == {"only"}
    assert second.entries() == {}


def test_registry_rejects_duplicate_name_and_args_type():
    registry = _RegistryA()
    registry.register("one", _Args, lambda args: None)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("one", _Args, lambda args: None)
    with pytest.raises(ValueError, match="Args type"):
        registry.register("two", _Args, lambda args: None)


def test_single_algorithm_registry_keeps_required_subcommand():
    registry = _RegistryA()
    registry.register("only", _Args, lambda args: None)
    registry._discovered = True

    parsed = registry.parse_args(["only", "--value", "7"])

    assert parsed == _Args(value=7)
    with pytest.raises(SystemExit):
        registry.parse_args(["--value", "7"])


def test_dispatch_uses_exact_args_type_for_inherited_configs():
    @dataclass
    class ChildArgs(_Args):
        pass

    calls = []
    registry = _RegistryA()
    registry.register("parent", _Args, lambda args: calls.append("parent"))
    registry.register("child", ChildArgs, lambda args: calls.append("child"))

    registry.dispatch(ChildArgs())

    assert calls == ["child"]


def test_phase_registries_discover_expected_algorithms():
    from rl_garden.training.off2on import registry as off2on
    from rl_garden.training.offline import registry as offline
    from rl_garden.training.online import registry as online

    online.discover()
    offline.discover()
    off2on.discover()

    assert set(online.entries()) == {
        "sac", "ppo", "recurrent_ppo", "recurrent_sac", "drqv2", "flash_sac", "residual_sac"
    }
    assert set(offline.entries()) == {"bc", "iql", "cql", "calql", "wsrl"}
    assert set(off2on.entries()) == {"wsrl"}


def test_environment_logging_defaults_are_overridden_by_explicit_cli(monkeypatch):
    from rl_garden.training.online import registry

    monkeypatch.setenv("RLG_LOG_TYPE", "wandb")

    args = registry.parse_args(["sac", "--log-type", "none"])

    assert args.log_type == "none"


def test_sac_disables_eval_env_when_eval_frequency_is_zero():
    from rl_garden.training.online.sac import SACArgs, _sac_env_request

    args = SACArgs(eval_freq=0)

    assert not _sac_env_request(args, "test-run").create_eval_env


def test_print_config_is_recursive_and_does_not_create_run_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"RLG_LOG_TYPE": "wandb", "MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_online.py",
            "sac",
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
    assert config["training_phase"] == "online"
    assert config["algorithm"] == "sac"
    assert config["args"]["log_type"] == "none"
    assert config["args"]["env_backend"] == "robotwin"
    assert isinstance(config["args"]["robotwin"], dict)
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_residual_sac_print_config_does_not_create_training_resources(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"RLG_LOG_TYPE": "wandb", "MPLCONFIGDIR": "/tmp"}
    result = subprocess.run(
        [
            sys.executable,
            "examples/train_online.py",
            "residual_sac",
            "--print-config",
            "--debug",
            "--log-type",
            "none",
            "--log-dir",
            str(tmp_path),
            "--maniskill.env-kwargs-json",
            '{"robot_uids": "panda_wristcam_gripper_closed", "fix_box": true}',
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(result.stdout)
    assert config["training_phase"] == "online"
    assert config["algorithm"] == "residual_sac"
    assert config["args"]["debug"] is True
    assert config["args"]["maniskill"]["env_kwargs_json"] == (
        '{"robot_uids": "panda_wristcam_gripper_closed", "fix_box": true}'
    )
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_help_does_not_import_simulator_backends():
    repo_root = Path(__file__).resolve().parents[1]
    command = """
from rl_garden.training.online import registry
registry.discover()
import sys
assert 'rl_garden.envs.backends.maniskill' not in sys.modules
assert 'rl_garden.envs.backends.robotwin' not in sys.modules
assert 'mani_skill' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        check=True,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
    )
