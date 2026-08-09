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


class _RequiresUnmappedParameter:
    def __init__(self, required):
        self.required = required


@dataclass
class _EmptyArgs:
    pass


def test_registry_instances_are_isolated_and_entries_are_copied():
    first = _RegistryA()
    second = _RegistryB()
    first.register("only", _Args, lambda args: None, contract_mode="passthrough")

    entries = first.entries()
    entries.clear()

    assert set(first.entries()) == {"only"}
    assert second.entries() == {}


def test_all_public_training_contracts_cover_fields_and_constructors():
    from rl_garden.common.effective_config import default_provenance
    from rl_garden.training.off2on import registry as off2on
    from rl_garden.training.offline import registry as offline
    from rl_garden.training.online import registry as online

    for phase_registry in (online, offline, off2on):
        phase_registry.discover()
        for entry in phase_registry.entries().values():
            provenance = default_provenance(entry.args_cls())
            applied = entry.contract.apply(provenance)
            assert set(applied) == set(provenance)
            assert all(field.owner != "unclassified" for field in applied.values())
            entry.contract.constructor_defaults()


def test_contract_rejects_unmapped_required_constructor_parameter():
    from rl_garden.training.config_contract import ConfigContract

    contract = ConfigContract.for_args(
        _EmptyArgs,
        target=f"{__name__}._RequiresUnmappedParameter",
    )

    with pytest.raises(ValueError, match="required.*not mapped"):
        contract.constructor_defaults()


def test_registry_rejects_duplicate_name_and_args_type():
    registry = _RegistryA()
    registry.register("one", _Args, lambda args: None, contract_mode="passthrough")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("one", _Args, lambda args: None, contract_mode="passthrough")
    with pytest.raises(ValueError, match="Args type"):
        registry.register("two", _Args, lambda args: None, contract_mode="passthrough")


def test_single_algorithm_registry_keeps_required_subcommand():
    registry = _RegistryA()
    registry.register("only", _Args, lambda args: None, contract_mode="passthrough")
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
    registry.register(
        "parent", _Args, lambda args: calls.append("parent"), contract_mode="passthrough"
    )
    registry.register(
        "child", ChildArgs, lambda args: calls.append("child"), contract_mode="passthrough"
    )

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
        "ppo",
        "recurrent_ppo",
        "recurrent_sac",
        "transformer_ppo",
        "transformer_sac",
        "drqv2",
        "flash_sac",
        "residual_sac",
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
    assert config["schema_version"] == 2
    assert config["status"] == "preflight"
    assert config["selection"] == {"training_phase": "online", "algorithm": "sac"}
    assert config["inputs"]["log_type"] == "none"
    assert config["inputs"]["env_backend"] == "robotwin"
    assert isinstance(config["inputs"]["robotwin"], dict)
    assert "maniskill" not in config["inputs"]
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
    assert config["selection"] == {
        "training_phase": "online",
        "algorithm": "residual_sac",
    }
    assert config["inputs"]["debug"] is True
    assert config["inputs"]["maniskill"]["env_kwargs_json"] == (
        '{"robot_uids": "panda_wristcam_gripper_closed", "fix_box": true}'
    )
    assert list(tmp_path.iterdir()) == []
    assert "mani_skill" not in result.stderr


def test_preset_environment_and_cli_precedence(tmp_path, monkeypatch):
    from rl_garden.training.online import registry

    preset = tmp_path / "sac.yaml"
    preset.write_text("gamma: 0.91\nlog_type: tensorboard\n", encoding="utf-8")
    monkeypatch.setenv("RLG_LOG_TYPE", "wandb")

    command = registry.parse_command(
        ["sac", "--config", str(preset), "--gamma", "0.97", "--log-type", "none"]
    )

    assert command.args.gamma == 0.97
    assert command.args.log_type == "none"
    assert [item.kind for item in command.provenance["gamma"].history][-2:] == [
        "preset",
        "CLI",
    ]
    assert [item.kind for item in command.provenance["log_type"].history][-3:] == [
        "preset",
        "RLG_*",
        "CLI",
    ]


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
        (online, "residual_sac", "configs/online/residual_sac_state.yaml"),
        (online, "residual_sac", "configs/online/residual_sac_rgb.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl_rgb.yaml"),
    ]

    for phase_registry, algorithm, relative_path in cases:
        command = phase_registry.parse_command(
            [algorithm, "--config", str(repo_root / relative_path)]
        )
        config = phase_registry._preflight_config(command)
        assert config.selection["algorithm"] == algorithm


def test_sac_launcher_selects_resnet_preset_and_preserves_cli_override():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "scripts/train_sac_rgbd.sh",
            "--encoder",
            "resnet18",
            "--gamma",
            "0.91",
            "--log-type",
            "none",
            "--print-config",
        ],
        cwd=repo_root,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
        check=True,
        capture_output=True,
        text=True,
    )

    config = json.loads(result.stdout)
    assert config["inputs"]["encoder"] == "resnet18"
    assert config["inputs"]["q_lr"] == 0.0001
    assert config["inputs"]["gamma"] == 0.91
    assert config["runtime"]["launcher"] == "scripts/train_sac_rgbd.sh"
    assert config["provenance"]["q_lr"]["source"]["kind"] == "launcher"


def test_launcher_cli_override_remains_cli_provenance():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "scripts/train_sac_rgbd.sh",
            "--encoder",
            "resnet18",
            "--q-lr",
            "0.002",
            "--explain-param",
            "q_lr",
        ],
        cwd=repo_root,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["value"] == 0.002
    assert payload["source"]["kind"] == "CLI"
    assert [item["kind"] for item in payload["history"]][-2:] == ["launcher", "CLI"]


def test_launcher_explicit_config_suppresses_launcher_default(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    preset = tmp_path / "custom.yaml"
    preset.write_text("obs_mode: state\ngamma: 0.77\n", encoding="utf-8")

    result = subprocess.run(
        [
            "scripts/train_sac_rgbd.sh",
            "--config",
            str(preset),
            "--print-config",
        ],
        cwd=repo_root,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["inputs"]["gamma"] == 0.77
    assert payload["provenance"]["gamma"]["source"]["kind"] == "preset"


def test_runtime_normalization_is_reflected_in_inputs_and_provenance(monkeypatch):
    from rl_garden.training.online import registry

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    command = registry.parse_command(["sac", "--buffer-device", "cuda"])
    config = registry._preflight_config(command)

    assert command.args.buffer_device == "cpu"
    assert config.inputs["buffer_device"] == "cpu"
    assert config.derived["buffer_device"]["before"] == "cuda"
    assert config.provenance["buffer_device"].source.kind == "runtime-derived"


def test_contract_rejects_agent_field_missing_from_constructor():
    from dataclasses import dataclass

    from rl_garden.training.config_contract import ConfigContract

    @dataclass
    class Args:
        forgotten_parameter: int = 1

    contract = ConfigContract.for_args(
        Args, target=f"{__name__}._RequiresUnmappedParameter"
    )

    with pytest.raises(ValueError, match="forgotten_parameter.*does not match"):
        contract.constructor_defaults()


def test_logging_checkpoint_visual_ownership_reflects_mixin_fields():
    from dataclasses import dataclass

    from rl_garden.common.cli_args import CheckpointArgs, LoggingArgs, VisionArgs
    from rl_garden.training.config_contract import ConfigContract

    @dataclass
    class Args(LoggingArgs, CheckpointArgs, VisionArgs):
        gamma: float = 0.99

    contract = ConfigContract.for_args(Args, target="example.Agent")

    for logging_field in {f.name for f in LoggingArgs.__dataclass_fields__.values()}:
        assert contract.fields[logging_field].owner == "logging"
    for checkpoint_field in {
        f.name for f in CheckpointArgs.__dataclass_fields__.values()
    }:
        assert contract.fields[checkpoint_field].owner == "checkpoint"
    assert contract.fields["gamma"].owner == "agent"
    assert contract.fields["encoder"].active_when.describe() == "obs_mode != 'state'"
    assert contract.fields["obs_mode"].active_when.describe() == "always"


def test_contract_mode_decoupled_from_target():
    from dataclasses import dataclass

    from rl_garden.training.config_contract import ConfigContract

    @dataclass
    class Args:
        forgotten_parameter: int = 1

    strict = ConfigContract.for_args(
        Args, target=f"{__name__}._RequiresUnmappedParameter", mode="strict"
    )
    assert strict.fields["forgotten_parameter"].owner == "agent"
    with pytest.raises(ValueError, match="forgotten_parameter.*does not match"):
        strict.constructor_defaults()

    passthrough = ConfigContract.for_args(Args, target="example.Agent", mode="passthrough")
    assert passthrough.mode == "passthrough"
    assert passthrough.fields["forgotten_parameter"].owner == "unused"


def test_registry_register_contract_mode_independent_of_target():
    # Omitting target= no longer implies passthrough -- contract_mode is the
    # only thing that controls it, and defaults to strict either way.
    registry = _RegistryA()
    registry.register("no_target_still_strict", _Args, lambda args: None)

    entry = registry.entries()["no_target_still_strict"]
    assert entry.contract.mode == "strict"
    assert entry.contract.fields["value"].owner == "agent"


def test_print_config_runs_constructor_contract_validation():
    registry = _RegistryA()
    registry.register(
        "broken",
        _Args,
        lambda args: None,
        target=f"{__name__}._RequiresUnmappedParameter",
    )
    registry._discovered = True

    with pytest.raises(SystemExit, match="Invalid config contract"):
        registry.run_cli(["broken", "--print-config"])


def test_constructor_introspection_walks_mro_and_skips_runtime_parameters():
    from rl_garden.training._constructor_introspection import (
        inspect_constructor_parameters,
    )

    class Parent:
        def __init__(self, env, inherited=1):
            pass

    class Child(Parent):
        def __init__(self, own=2, **kwargs):
            pass

    parameters = inspect_constructor_parameters(Child)

    assert list(parameters) == ["own", "inherited"]


def test_constructor_introspection_skips_uninspectable_mro_entry(monkeypatch):
    import rl_garden.training._constructor_introspection as constructor_introspection

    class Parent:
        def __init__(self, inherited=1):
            pass

    class Child(Parent):
        def __init__(self, own=2):
            pass

    original_signature = constructor_introspection.inspect.signature

    def signature(init):
        if init is Child.__init__:
            raise ValueError("uninspectable")
        return original_signature(init)

    monkeypatch.setattr(constructor_introspection.inspect, "signature", signature)

    assert list(constructor_introspection.inspect_constructor_parameters(Child)) == [
        "inherited"
    ]


def test_real_world_algorithms_are_passthrough_mode():
    from rl_garden.training.real_world._registry import registry

    registry.discover()
    for name in ("serl", "hil_serl"):
        assert registry.entries()[name].contract.mode == "passthrough"


def test_print_config_reports_contract_mode(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(
        ["sac", "--obs-mode", "state", "--log-type", "none", "--print-config"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"]["mode"] == "strict"


def test_sac_contract_exposes_transformed_network_mapping():
    from rl_garden.training.online import registry

    registry.discover()
    contract = registry.entries()["sac"].contract

    assert contract.fields["actor_hidden_layers"].mapped_to.endswith(".net_arch.pi")
    assert contract.fields["critic_hidden_layers"].mapped_to.endswith(".net_arch.qf")


def test_active_condition_describe_matches_prior_string_format():
    from rl_garden.training.config_contract import ActiveCondition

    cases = [
        (ActiveCondition("always"), "always"),
        (ActiveCondition("env_backend_is", "maniskill"), "env_backend == 'maniskill'"),
        (ActiveCondition("visual_observation"), "obs_mode != 'state'"),
        (
            ActiveCondition("encoder_starts_with", "resnet"),
            "obs_mode != 'state' and encoder starts with 'resnet'",
        ),
        (
            ActiveCondition("encoder_is", "plain_conv"),
            "obs_mode != 'state' and encoder == 'plain_conv'",
        ),
        (
            ActiveCondition("encoder_is", "vit"),
            "obs_mode != 'state' and encoder == 'vit'",
        ),
    ]
    for condition, expected in cases:
        assert condition.describe() == expected


def test_active_condition_evaluate_matches_prior_semantics():
    from dataclasses import dataclass

    from rl_garden.training.config_contract import ActiveCondition

    @dataclass
    class FakeArgs:
        env_backend: str = "maniskill"
        obs_mode: str = "rgb"
        encoder: str = "resnet18"

    args = FakeArgs()
    assert ActiveCondition("always").evaluate(args) is True
    assert ActiveCondition("env_backend_is", "maniskill").evaluate(args) is True
    assert ActiveCondition("env_backend_is", "robotwin").evaluate(args) is False
    assert ActiveCondition("visual_observation").evaluate(args) is True
    assert ActiveCondition("encoder_starts_with", "resnet").evaluate(args) is True
    assert ActiveCondition("encoder_is", "plain_conv").evaluate(args) is False

    state_args = FakeArgs(obs_mode="state")
    assert ActiveCondition("visual_observation").evaluate(state_args) is False
    assert ActiveCondition("encoder_starts_with", "resnet").evaluate(state_args) is False


def test_active_condition_fails_closed_on_unrecognized_kind():
    from rl_garden.training.config_contract import ActiveCondition

    with pytest.raises(ValueError, match="Unknown ActiveCondition kind"):
        ActiveCondition(kind="something_new")  # type: ignore[arg-type]


def test_active_condition_validates_value_shape():
    from rl_garden.training.config_contract import ActiveCondition

    with pytest.raises(ValueError, match="requires a value"):
        ActiveCondition("encoder_is")
    with pytest.raises(ValueError, match="does not accept a value"):
        ActiveCondition("always", "unused")


def test_print_config_does_not_expose_internal_consumption(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(
        ["sac", "--obs-mode", "state", "--log-type", "none", "--print-config"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert "consumption" not in payload["algorithm"]


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

    with pytest.raises(SystemExit, match="offline_dataset_path is required"):
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


def test_explain_param_prints_machine_readable_provenance(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(["sac", "--gamma", "0.93", "--explain-param", "gamma"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == "gamma"
    assert payload["value"] == 0.93
    assert payload["type"] == "float"
    assert payload["source"]["kind"] == "CLI"
    assert payload["defined_at"].endswith("rl_garden/training/online/_args.py:21")


def test_dry_run_materializes_env_and_agent_without_learning(monkeypatch, capsys):
    import gymnasium as gym

    from rl_garden.training import inspection
    from rl_garden.training.online import _runner as online_runner
    from rl_garden.training.online import registry
    from rl_garden.training.online import sac as sac_module

    monkeypatch.setattr(
        inspection, "validate_constructor_coverage", lambda *args, **kwargs: None
    )

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


def test_sac_dry_run_captures_exact_constructor_kwargs(monkeypatch, capsys):
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
            "0",
            "--log-type",
            "none",
        ]
    )

    algorithm = json.loads(capsys.readouterr().out)["algorithm"]
    assert algorithm["target"] == "rl_garden.algorithms.sac.SAC"
    assert algorithm["constructor_kwargs"]["net_arch"] == {
        "pi": [8, 8],
        "qf": [8],
    }
    assert algorithm["constructor_kwargs"]["gamma"] == 0.8
    assert algorithm["field_mappings"]["actor_hidden_layers"].endswith(".net_arch.pi")
    assert "gamma" not in algorithm["implicit_defaults"]


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


def test_real_world_registry_rejects_dry_run_via_single_parse(monkeypatch):
    from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry
    from rl_garden.training.real_world._registry import registry

    assert registry.supports_dry_run is False

    call_count = 0
    original_parse_command = BaseAlgorithmRegistry.parse_command

    def counting_parse_command(self, args=None):
        nonlocal call_count
        call_count += 1
        return original_parse_command(self, args)

    monkeypatch.setattr(
        BaseAlgorithmRegistry, "parse_command", counting_parse_command
    )

    with pytest.raises(SystemExit, match="not supported"):
        registry.run_cli(["serl", "--dry-run", "--log-type", "none"])

    assert call_count == 1


def test_check_constructor_coverage_direct_kwarg():
    from rl_garden.training.config_contract import (
        DirectKwarg,
        check_constructor_coverage,
    )

    ok = check_constructor_coverage(
        {"gamma": DirectKwarg("gamma")}, {"gamma": 0.99}
    )
    assert ok == []

    missing = check_constructor_coverage(
        {"gamma": DirectKwarg("gamma")}, {"tau": 0.01}
    )
    assert len(missing) == 1
    assert "'gamma'" in missing[0]


def test_check_constructor_coverage_nested_kwarg():
    from rl_garden.training.config_contract import (
        NestedKwarg,
        check_constructor_coverage,
    )

    ok = check_constructor_coverage(
        {"actor_hidden_layers": NestedKwarg("net_arch", "pi")},
        {"net_arch": {"pi": [256, 256], "qf": [256, 256]}},
    )
    assert ok == []

    missing_container = check_constructor_coverage(
        {"actor_hidden_layers": NestedKwarg("net_arch", "pi")}, {}
    )
    assert len(missing_container) == 1

    missing_key = check_constructor_coverage(
        {"actor_hidden_layers": NestedKwarg("net_arch", "pi")},
        {"net_arch": {"qf": [256, 256]}},
    )
    assert len(missing_key) == 1


def test_check_constructor_coverage_builder_derived_is_presence_only():
    from rl_garden.training.config_contract import (
        BuilderDerived,
        check_constructor_coverage,
    )

    # Any single cluster kwarg present satisfies every field mapped to that
    # cluster -- this is deliberately coarse, not a per-field value check.
    ok = check_constructor_coverage(
        {
            "encoder": BuilderDerived("visual_encoder"),
            "pretrained_weights": BuilderDerived("visual_encoder"),
        },
        {"image_encoder_factory": object()},
    )
    assert ok == []

    missing = check_constructor_coverage(
        {"encoder": BuilderDerived("visual_encoder")}, {"unrelated_kwarg": 1}
    )
    assert len(missing) == 1

    inactive = check_constructor_coverage(
        {"encoder": BuilderDerived("visual_encoder")},
        {},
        inactive_clusters=frozenset({"visual_encoder"}),
    )
    assert inactive == []


def test_check_constructor_coverage_unknown_consumption_type_raises():
    from rl_garden.training.config_contract import (
        BuilderDerived,
        check_constructor_coverage,
    )

    with pytest.raises(TypeError, match="Unhandled consumption type"):
        check_constructor_coverage({"gamma": "not-a-consumption-object"}, {})
    with pytest.raises(ValueError, match="Unknown builder-derived consumption cluster"):
        check_constructor_coverage({"encoder": BuilderDerived("missing")}, {})


def test_sac_dry_run_with_deliberately_dropped_kwarg_fails(monkeypatch, capsys):
    """A strict contract rejects a builder that drops a declared field."""
    from rl_garden.algorithms import SAC
    from rl_garden.training.inspection import construct_agent
    from rl_garden.training.online import _runner as online_runner
    from rl_garden.training.online import registry
    from rl_garden.training.online import sac as sac_module

    class FakeEnv:
        single_observation_space = __import__("gymnasium").spaces.Box(
            -1.0, 1.0, shape=(3,)
        )
        single_action_space = __import__("gymnasium").spaces.Box(
            -1.0, 1.0, shape=(2,)
        )
        num_envs = 1

        def close(self):
            pass

    monkeypatch.setattr(
        online_runner, "make_training_envs", lambda backend, req: (FakeEnv(), None)
    )

    def dropping_build_sac(args, env, eval_env, logger, checkpoint_dir):
        kwargs = sac_module._sac_common_kwargs(
            args, env, eval_env, logger, checkpoint_dir, {}
        )
        del kwargs["gamma"]  # simulate a builder that forgot to pass gamma
        return construct_agent(SAC, **kwargs)

    monkeypatch.setattr(sac_module, "build_sac", dropping_build_sac)

    with pytest.raises(SystemExit, match=r"'gamma'.*not found"):
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
                "--log-type",
                "none",
            ]
        )
    capsys.readouterr()
