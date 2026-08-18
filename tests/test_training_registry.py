import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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
        "sac",
        "sac_flow",
        "ppo",
        "recurrent_ppo",
        "recurrent_sac",
        "transformer_ppo",
        "transformer_sac",
        "drqv2",
        "flash_sac",
        "td3",
        "rlpd",
        "rlpd_hybrid",
        "tdmpc2",
    }
    assert set(offline.entries()) == {
        "bc",
        "iql",
        "cql",
        "calql",
        "wsrl",
        "tdmpc2_multitask",
        "td3_bc",
        "awac",
    }
    assert set(off2on.entries()) == {"wsrl", "calql", "iql", "awac"}


def test_logging_environment_variables_are_not_configuration(monkeypatch):
    from rl_garden.training.online import registry

    monkeypatch.setenv("RLG_LOG_TYPE", "none")

    args = registry.parse_args(["sac"])

    assert args.log_type == "wandb"


def test_sac_disables_eval_env_when_eval_frequency_is_zero():
    from rl_garden.training.online.sac import SACArgs, _sac_env_request

    args = SACArgs(eval_freq=0)

    assert not _sac_env_request(args, "test-run").create_eval_env


def test_print_config_is_recursive_and_does_not_create_run_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {"MPLCONFIGDIR": "/tmp"}
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
    assert config["schema_version"] == 3
    assert config["status"] == "preflight"
    assert config["selection"] == {"training_phase": "online", "algorithm": "sac"}
    assert config["inputs"]["log_type"] == "none"
    assert config["inputs"]["env_backend"] == "robotwin"
    assert "robotwin" not in config["inputs"]
    assert "maniskill" not in config["inputs"]
    assert config["active_environment"]["backend"] == "robotwin"
    assert isinstance(config["active_environment"]["config"], dict)
    assert config["algorithm"] == {}
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_preset_and_cli_precedence(tmp_path):
    from rl_garden.training.online import registry

    preset = tmp_path / "sac.yaml"
    preset.write_text("gamma: 0.91\nlog_type: tensorboard\n", encoding="utf-8")
    command = registry.parse_command(
        ["sac", "--config", str(preset), "--gamma", "0.97", "--log-type", "none"]
    )

    assert command.args.gamma == 0.97
    assert command.args.log_type == "none"
    assert command.sources["gamma"].kind == "CLI"
    assert command.sources["log_type"].kind == "CLI"


def test_command_rejects_multiple_presets(tmp_path):
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("gamma: 0.91\n", encoding="utf-8")
    second.write_text("gamma: 0.92\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Only one --config preset"):
        registry.parse_command(
            ["sac", "--config", str(first), "--config", str(second)]
        )


def test_checked_in_presets_pass_static_preflight():
    from rl_garden.training.off2on import registry as off2on
    from rl_garden.training.online import registry as online

    repo_root = Path(__file__).resolve().parents[1]
    cases = [
        (online, "sac", "configs/online/sac_state.yaml"),
        (online, "sac", "configs/online/sac_rgb.yaml"),
        (online, "sac", "configs/online/sac_rgb_resnet.yaml"),
        (online, "ppo", "configs/online/ppo_state.yaml"),
        (online, "ppo", "configs/online/ppo_rgb.yaml"),
        (online, "ppo", "configs/online/ppo_robotwin_place_empty_cup_rgb.yaml"),
        (online, "drqv2", "configs/online/drqv2_rgb.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl_rgb.yaml"),
    ]

    for phase_registry, algorithm, relative_path in cases:
        command = phase_registry.parse_command(
            [algorithm, "--config", str(repo_root / relative_path)]
        )
        config = phase_registry._preflight_config(command)
        assert config.selection["algorithm"] == algorithm


def test_robotwin_launcher_keeps_dependency_env_and_forwards_cli_override():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ | {
        "MPLCONFIGDIR": "/tmp",
        "RLG_ROBOTWIN_ROOT": "/tmp/robotwin",
        "RLG_ROBOTWIN_ASSETS_PATH": "/tmp/default-assets",
    }
    result = subprocess.run(
        [
            "scripts/train_ppo_robotwin_place_empty_cup_rgbd.sh",
            "--robotwin.assets-path",
            "/tmp/cli-assets",
            "--log-type",
            "none",
            "--print-config",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(result.stdout)
    assert config["active_environment"]["config"]["robotwin_root"] == (
        "/tmp/robotwin"
    )
    assert config["active_environment"]["config"]["assets_path"] == (
        "/tmp/cli-assets"
    )
    assert config["sources"]["robotwin.assets_path"]["kind"] == "CLI"


def test_runtime_normalization_is_reflected_in_inputs_and_sources(monkeypatch):
    from rl_garden.training.online import registry

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    command = registry.parse_command(["sac", "--buffer-device", "cuda"])
    config = registry._preflight_config(command)

    assert command.args.buffer_device == "cpu"
    assert config.inputs["buffer_device"] == "cpu"
    assert config.derived["buffer_device"]["before"] == "cuda"
    assert config.sources["buffer_device"].kind == "runtime-derived"


def test_print_config_has_no_static_algorithm_inference(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(
        ["sac", "--obs-mode", "state", "--log-type", "none", "--print-config"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["algorithm"] == {}


def test_preflight_rejects_unknown_backend_and_invalid_backend_json():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    unknown = registry.parse_command(["sac", "--env-backend", "missing"])
    with pytest.raises(ConfigError, match="Unknown env backend 'missing'"):
        registry._preflight_config(unknown)

    invalid_json = registry.parse_command(
        ["sac", "--maniskill.env-kwargs-json", "not-json"]
    )
    with pytest.raises(ConfigError, match="Invalid maniskill.env_kwargs_json"):
        registry._preflight_config(invalid_json)

    non_object_json = registry.parse_command(
        ["sac", "--maniskill.env-kwargs-json", "[]"]
    )
    with pytest.raises(ConfigError, match="Invalid maniskill.env_kwargs_json"):
        registry._preflight_config(non_object_json)


def test_preflight_rejects_explicit_inactive_backend_field():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    command = registry.parse_command(
        ["sac", "--env-backend", "maniskill", "--robotwin.step-lim", "12"]
    )

    with pytest.raises(ConfigError, match="robotwin.step_lim.*inactive"):
        registry._preflight_config(command)


def test_preflight_rejects_explicit_inactive_visual_field():
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    command = registry.parse_command(["sac", "--obs-mode", "state", "--encoder", "vit"])

    with pytest.raises(ConfigError, match="encoder.*inactive"):
        registry._preflight_config(command)


def test_off2on_dry_run_validates_dataset_before_creating_resources(monkeypatch):
    from rl_garden.training.off2on import _runner, registry

    monkeypatch.setattr(
        _runner,
        "make_training_envs",
        lambda *args: (_ for _ in ()).throw(AssertionError("env must not be created")),
    )

    with pytest.raises(SystemExit, match="offline_dataset is required"):
        registry.run_cli(
            [
                "wsrl",
                "--dry-run",
                "--num-offline-steps",
                "1",
                "--log-type",
                "none",
            ]
        )


def test_explain_param_prints_only_value_and_source(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(["sac", "--gamma", "0.93", "--explain-param", "gamma"])

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"path", "value", "type", "source"}
    assert payload["path"] == "gamma"
    assert payload["value"] == 0.93
    assert payload["type"] == "float"
    assert payload["source"]["kind"] == "CLI"


def test_explain_param_reports_default_source(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(["sac", "--explain-param", "gamma"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == {"kind": "default", "detail": None}


def test_explain_param_rejects_inactive_field():
    from rl_garden.training.online import registry

    with pytest.raises(SystemExit, match="encoder.*inactive"):
        registry.run_cli(["sac", "--obs-mode", "state", "--explain-param", "encoder"])


def test_dry_run_materializes_env_and_agent_without_learning(monkeypatch, capsys):
    import gymnasium as gym

    from rl_garden.training.online import _runner as online_runner
    from rl_garden.training.online import registry
    from rl_garden.training.online import sac as sac_module

    class FakeEnv:
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        num_envs = 1
        closed = False

        def close(self):
            self.closed = True

    class FakeAgent:
        device = "cpu"
        gamma = 0.8
        steps_per_env = 4
        grad_steps_per_iteration = 2

        def learn(self, **kwargs):
            raise AssertionError("dry-run must not call learn()")

    env = FakeEnv()
    captured = {}
    monkeypatch.setattr(
        online_runner,
        "make_training_envs",
        lambda backend, req: (env, None),
    )

    def fake_build(args, train_env, eval_env, logger, checkpoint_dir):
        from rl_garden.training.inspection import construct_agent

        captured.update(
            args=args,
            env=train_env,
            eval_env=eval_env,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
        )
        return construct_agent(FakeAgent)

    monkeypatch.setattr(sac_module, "build_sac", fake_build)

    registry.run_cli(
        [
            "sac",
            "--dry-run",
            "--obs-mode",
            "state",
            "--buffer-device",
            "cpu",
            "--eval-freq",
            "0",
            "--no-save-final-checkpoint",
            "--log-type",
            "none",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "materialized"
    assert payload["algorithm"]["target"].endswith("FakeAgent")
    assert payload["algorithm"]["constructor_kwargs"] == {}
    assert payload["runtime"]["dry_run"] is True
    assert captured["logger"].log_type == "none"
    assert captured["checkpoint_dir"] is None
    assert env.closed


def test_sac_dry_run_captures_exact_constructor_kwargs(
    monkeypatch, capsys, tmp_path
):
    import gymnasium as gym

    from rl_garden.training.online import _runner as online_runner
    from rl_garden.training.online import registry

    class FakeEnv:
        num_envs = 1
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    monkeypatch.setattr(
        online_runner,
        "make_training_envs",
        lambda backend, req: (FakeEnv(), None),
    )

    registry.run_cli(
        [
            "sac",
            "--dry-run",
            "--obs-mode",
            "state",
            "--buffer-device",
            "cpu",
            "--buffer-size",
            "16",
            "--batch-size",
            "2",
            "--hidden-dim",
            "8",
            "--actor-hidden-layers",
            "2",
            "--critic-hidden-layers",
            "1",
            "--eval-freq",
            "99",
            "--checkpoint-freq",
            "12",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--log-type",
            "none",
        ]
    )

    algorithm = json.loads(capsys.readouterr().out)["algorithm"]
    assert set(algorithm) == {"target", "constructor_kwargs"}
    assert algorithm["target"] == "rl_garden.algorithms.sac.SAC"
    assert algorithm["constructor_kwargs"]["net_arch"] == {
        "pi": [8, 8],
        "qf": [8],
    }
    assert algorithm["constructor_kwargs"]["gamma"] == 0.8
    assert algorithm["constructor_kwargs"]["eval_freq"] == 99
    assert algorithm["constructor_kwargs"]["checkpoint_freq"] == 12


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


def test_backend_discovery_validates_v2_without_importing_simulators():
    repo_root = Path(__file__).resolve().parents[1]
    command = """
from rl_garden.envs.backend_registry import _REGISTRY, discover_env_backends
discover_env_backends()
import sys
assert _REGISTRY
assert all(backend.api_version == 2 for backend in _REGISTRY.values())
assert 'mani_skill' not in sys.modules
assert 'sapien' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        check=True,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
    )


def test_help_lists_config_inspection_controls(capsys):
    from rl_garden.training.online import registry

    with pytest.raises(SystemExit) as exc_info:
        registry.run_cli(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--config PATH" in output
    assert "--print-config" in output
    assert "--dry-run" in output
    assert "--explain-param PATH" in output
