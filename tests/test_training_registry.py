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
        "dppo",
        "diffusion_cm_distill_online",
        "flow_ppo",
        "recurrent_ppo",
        "recurrent_sac",
        "transformer_ppo",
        "transformer_sac",
        "drqv2",
        "flash_sac",
        "td3",
        "rlpd",
        "rlpd_hybrid",
        "explore",
        "supe",
        "jsrl",
        "acrlpd",
        "tdmpc2",
        "dreamer_v3",
        "ncrl",
        "dagger",
        "policy_distillation",
        "gail",
    }
    assert set(offline.entries()) == {
        "a2a_bc",
        "bc",
        "diffusion_bc",
        "consistency_distill_bc",
        "flow_bc",
        "mean_flow_bc",
        "fql",
        "floq",
        "value_flows",
        "fino",
        "fac",
        "hilp",
        "idql",
        "opal",
        "iql",
        "cql",
        "calql",
        "wsrl",
        "tdmpc2_multitask",
        "pwm",
        "td3_bc",
        "awac",
        "qgf",
        "qam",
        "rebrac",
        "edac",
        "spot",
        "bcq",
        "plas",
        "bppo",
        "unio4",
        "unio4_ope",
        "rssm_pretrain",
    }
    assert set(off2on.entries()) == {
        "wsrl",
        "calql",
        "iql",
        "awac",
        "acfql",
        "floq",
        "value_flows",
        "fino",
        "fql",
        "fac",
        "spot",
        "so2",
    }


def test_logging_environment_variables_are_not_configuration(monkeypatch):
    from rl_garden.training.online import registry

    monkeypatch.setenv("RLG_LOG_TYPE", "none")

    args = registry.parse_args(["sac"])

    assert args.log_type == "tensorboard"


def test_sac_disables_eval_env_when_eval_frequency_is_zero():
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online.entrypoints.sac import SACArgs

    args = SACArgs(eval_freq=0)

    assert not make_env_request(args, "test-run").create_eval_env


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
        (online, "drqv2", "configs/online/drqv2_rgbd.yaml"),
        (online, "td3", "configs/online/td3_rgbd.yaml"),
        (online, "tdmpc2", "configs/online/tdmpc2_rgb.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl.yaml"),
        (off2on, "wsrl", "configs/off2on/wsrl_rgb.yaml"),
    ]

    for phase_registry, algorithm, relative_path in cases:
        command = phase_registry.parse_command(
            [algorithm, "--config", str(repo_root / relative_path)]
        )
        config = phase_registry._preflight_config(command)
        assert config.selection["algorithm"] == algorithm


def test_train_off2on_script_launches():
    """Canary: examples/train_off2on.py starts as a script (train_online and
    pretrain_offline are covered by the subprocess tests above and in
    test_cql_calql.py); every other off2on CLI check runs in-process."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/train_off2on.py", "--help"],
        cwd=repo_root,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--print-config" in result.stdout


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

    registry.run_cli(["sac", "--log-type", "none", "--print-config"])
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


@pytest.fixture
def external_isaaclab_backend():
    """An external (entry-point style) backend that reuses the stock --isaaclab.* section."""
    from rl_garden.envs.backend_registry import (
        _REGISTRY,
        EnvBackend,
        discover_env_backends,
        register_env_backend,
    )

    class ExternalIsaacLabBackend(EnvBackend):
        api_version = 2
        config_field = "external_isaaclab"

        @classmethod
        def config_from_args(cls, args):
            return args.isaaclab

        @classmethod
        def resolve_config(cls, req, *, is_eval):
            raise NotImplementedError

        @classmethod
        def make_train_env(cls, req):
            raise NotImplementedError

        @classmethod
        def make_eval_env(cls, req):
            raise NotImplementedError

    discover_env_backends()
    register_env_backend("external_isaaclab", ExternalIsaacLabBackend)
    yield "external_isaaclab"
    _REGISTRY.pop("external_isaaclab")


def test_preflight_accepts_registry_backend_without_own_cli_section(external_isaaclab_backend):
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    command = registry.parse_command(
        ["sac", "--env-backend", external_isaaclab_backend, "--isaaclab.sim-device", "cuda:1"]
    )
    config = registry._preflight_config(command)

    assert config.active_environment["backend"] == external_isaaclab_backend
    assert config.active_environment["config"]["sim_device"] == "cuda:1"

    # The reused section is the active one, so every other backend section stays inactive.
    other_section = registry.parse_command(
        ["sac", "--env-backend", external_isaaclab_backend, "--maniskill.sim-backend", "cpu"]
    )
    with pytest.raises(ConfigError, match="maniskill.sim_backend.*inactive"):
        registry._preflight_config(other_section)

    # Validation messages name the section the user typed.
    invalid_json = registry.parse_command(
        ["sac", "--env-backend", external_isaaclab_backend, "--isaaclab.env-kwargs-json", "[]"]
    )
    with pytest.raises(ConfigError, match="Invalid isaaclab.env_kwargs_json"):
        registry._preflight_config(invalid_json)


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

    # State-only by default (no --obs.rgb/--obs.depth); --encoder.backbone is
    # inactive without a visual --obs.
    command = registry.parse_command(["sac", "--encoder.backbone", "vit"])

    with pytest.raises(ConfigError, match="encoder.backbone.*inactive"):
        registry._preflight_config(command)


def test_off2on_dry_run_validates_dataset_before_creating_resources(monkeypatch):
    from rl_garden.training.off2on import registry, runner

    monkeypatch.setattr(
        runner,
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

    with pytest.raises(SystemExit, match="encoder.backbone.*inactive"):
        registry.run_cli(["sac", "--explain-param", "encoder.backbone"])


def test_dry_run_materializes_env_and_agent_without_learning(monkeypatch, capsys):
    import gymnasium as gym

    from rl_garden.training.online import registry
    from rl_garden.training.online import runner as online_runner
    from rl_garden.training.online.entrypoints import sac as sac_module

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

    from rl_garden.training.online import registry
    from rl_garden.training.online import runner as online_runner

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


# --- extension contract: entry points, declared builders, builder reuse --------------


class _FakeEntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.value = f"external_pkg.{name}"
        self.load = load


def _registry_over_empty_package(tmp_path, monkeypatch, name):
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    class _ExternalRegistry(BaseAlgorithmRegistry):
        package_name = name
        phase_name = "external"

    return _ExternalRegistry()


def _patch_algorithm_entry_points(monkeypatch, entry_points):
    from rl_garden.training import algorithm_registry

    def fake_entry_points(*, group):
        assert group == "rlgarden.algorithms"
        return entry_points

    monkeypatch.setattr(algorithm_registry, "_ALGORITHM_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(algorithm_registry.importlib.metadata, "entry_points", fake_entry_points)


def test_algorithm_entry_point_registers_an_algorithm_the_cli_finds(tmp_path, monkeypatch):
    registry = _registry_over_empty_package(tmp_path, monkeypatch, "ext_entrypoints_ok")
    loads = []

    def load():
        loads.append(1)
        registry.register("from_entry_point", _Args, lambda args: None)

    _patch_algorithm_entry_points(monkeypatch, [_FakeEntryPoint("ext", load)])

    command = registry.parse_command(["from_entry_point", "--value", "3"])

    assert command.algorithm == "from_entry_point"
    assert command.args == _Args(value=3)
    # Loaded once per process, not once per registry.
    _registry_over_empty_package(tmp_path, monkeypatch, "ext_entrypoints_other").discover()
    assert loads == [1]


def test_failing_algorithm_entry_point_is_reported_by_name(tmp_path, monkeypatch):
    from rl_garden.training import algorithm_registry

    registry = _registry_over_empty_package(tmp_path, monkeypatch, "ext_entrypoints_broken")

    def load():
        raise ImportError("No module named 'external_pkg'")

    _patch_algorithm_entry_points(monkeypatch, [_FakeEntryPoint("broken_ext", load)])

    with pytest.raises(RuntimeError, match="'broken_ext'.*No module named 'external_pkg'"):
        registry.discover()
    assert algorithm_registry._ALGORITHM_ENTRY_POINTS_LOADED is False
    with pytest.raises(RuntimeError, match="'broken_ext'"):
        registry.parse_command(["anything"])


@pytest.mark.parametrize("phase", ["online", "offline", "off2on"])
def test_every_in_tree_entry_declares_its_module_builder(phase):
    import importlib

    registry = importlib.import_module(f"rl_garden.training.{phase}.registry").registry
    registry.discover()
    for name, entry in registry.entries().items():
        module = importlib.import_module(entry.run_fn.__module__)
        if not module.__name__.startswith("rl_garden."):
            continue
        builders = [
            value
            for key, value in vars(module).items()
            if key.startswith("build_")
            and callable(value)
            and getattr(value, "__module__", None) == module.__name__
        ]
        assert len(builders) <= 1, name
        expected = builders[0] if builders else None
        assert entry.build_agent is expected, name


class _StubAgent:
    pass


_REUSABLE_BUILDERS = [
    ("ppo", "ppo", "build_ppo"),
    ("recurrent_ppo", "recurrent_ppo", "build_recurrent_ppo"),
    ("transformer_ppo", "transformer_ppo", "build_transformer_ppo"),
    ("gail", "gail", "build_gail"),
    ("sac", "sac", "build_sac"),
    ("recurrent_sac", "recurrent_sac", "build_recurrent_sac"),
    ("transformer_sac", "transformer_sac", "build_transformer_sac"),
]


@pytest.mark.parametrize("algorithm,module_name,builder_name", _REUSABLE_BUILDERS)
def test_builders_take_a_subclass_and_forward_extra_kwargs(
    monkeypatch, algorithm, module_name, builder_name
):
    import importlib

    from rl_garden.training import inspection
    from rl_garden.training.online import registry

    registry.discover()
    module = importlib.import_module(f"rl_garden.training.online.entrypoints.{module_name}")
    builder = getattr(module, builder_name)
    constructed = []
    monkeypatch.setattr(
        inspection,
        "construct_agent",
        lambda cls, **kwargs: constructed.append((cls, kwargs)) or _StubAgent(),
    )

    class Subclass:
        pass

    args = registry.entries()[algorithm].args_cls()
    builder(args, "env", "eval_env", "logger", "checkpoints")
    builder(args, "env", "eval_env", "logger", "checkpoints", algorithm_cls=Subclass, extra_flag=7)

    (default_cls, default_kwargs), (subclass, subclass_kwargs) = constructed
    assert default_cls is registry.entries()[algorithm].algorithm_cls()
    assert subclass is Subclass
    assert subclass_kwargs == {**default_kwargs, "extra_flag": 7}
    assert list(subclass_kwargs)[:-1] == list(default_kwargs)


_RUN_FUNCTIONS = [
    ("ppo", "run_ppo", "build_ppo"),
    ("recurrent_ppo", "run_recurrent_ppo", "build_recurrent_ppo"),
    ("transformer_ppo", "run_transformer_ppo", "build_transformer_ppo"),
    ("gail", "run_gail", "build_gail"),
    ("sac", "run_sac", "build_sac"),
    ("recurrent_sac", "run_recurrent_sac", "build_recurrent_sac"),
    ("transformer_sac", "run_transformer_sac", "build_transformer_sac"),
]


@pytest.mark.parametrize("module_name,run_name,builder_name", _RUN_FUNCTIONS)
def test_run_functions_train_with_a_given_builder(
    monkeypatch, module_name, run_name, builder_name
):
    import importlib

    from rl_garden.training.online import registry
    from rl_garden.training.online import runner as online_runner

    registry.discover()
    module = importlib.import_module(f"rl_garden.training.online.entrypoints.{module_name}")
    used = []
    monkeypatch.setattr(
        online_runner, "run_online", lambda args, **kwargs: used.append(kwargs["build_agent"])
    )
    args = registry.entries()[module_name].args_cls()

    def downstream_builder(*args, **kwargs):
        raise AssertionError("not called")

    getattr(module, run_name)(args)
    getattr(module, run_name)(args, build_agent=downstream_builder)

    assert used == [getattr(module, builder_name), downstream_builder]


def test_console_scripts_dispatch_to_the_phase_registries_and_evaluation(monkeypatch):
    import re

    from rl_garden.training import cli, evaluation
    from rl_garden.training.off2on import registry as off2on
    from rl_garden.training.offline import registry as offline
    from rl_garden.training.online import registry as online

    calls = []
    for name, registry in (("online", online), ("offline", offline), ("off2on", off2on)):
        monkeypatch.setattr(registry, "run_cli", lambda name=name: calls.append(name))
    monkeypatch.setattr(evaluation, "main", lambda: calls.append("eval"))

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    scripts = dict(re.findall(r'^(rl-garden-[\w-]+) = "rl_garden\.training\.cli:(\w+)"$', pyproject, re.M))
    assert scripts == {
        "rl-garden-train-online": "train_online",
        "rl-garden-train-offline": "train_offline",
        "rl-garden-train-off2on": "train_off2on",
        "rl-garden-eval": "eval_checkpoint",
    }
    for function_name in scripts.values():
        getattr(cli, function_name)()

    assert calls == ["online", "offline", "off2on", "eval"]
