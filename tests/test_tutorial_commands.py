"""Keep quickstart commands valid without constructing simulators."""
import json

import pytest
import tyro

from tests._cli import run_entry
from rl_garden.training.evaluation import EvalCheckpointArgs

QUICKSTART_COMMANDS = [
    [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--gymnasium.device", "cpu",
        "--buffer-device", "cpu", "--total-timesteps", "2000",
        "--learning-starts", "100", "--training-freq", "4", "--batch-size", "64",
        "--buffer-size", "10000", "--hidden-dim", "64", "--gamma", "0.99",
        "--eval-freq", "0", "--log-freq", "500", "--no-capture-video",
        "--exp-name", "sac-pendulum",
    ],
    [
        "sac", "--env-backend", "maniskill", "--env-id", "PushCube-v1",
        "--num-envs", "16", "--num-eval-envs", "1", "--buffer-device", "cuda",
        "--total-timesteps", "2000", "--learning-starts", "100",
        "--eval-freq", "0", "--no-capture-video", "--exp-name", "sac-pushcube",
    ],
]
EVALUATION_COMMAND = [
    "--checkpoint-path", "runs/sac-pendulum/checkpoints/final.pt",
    "--device", "cpu", "--num-eval-envs", "1", "--num-eval-episodes", "5",
]


@pytest.mark.parametrize("argv", QUICKSTART_COMMANDS, ids=["pendulum", "pushcube"])
def test_quickstart_print_config(argv: list[str]) -> None:
    result = run_entry("train_online", [*argv, "--print-config"])
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["selection"] == {"algorithm": "sac", "training_phase": "online"}
    assert config["inputs"]["total_timesteps"] == 2000


def test_quickstart_evaluation_arguments() -> None:
    # Evaluation has no phase-registry print-config action; parse its public args.
    args = tyro.cli(EvalCheckpointArgs, args=EVALUATION_COMMAND)
    assert args.checkpoint_path == "runs/sac-pendulum/checkpoints/final.pt"
    assert args.device == "cpu"
    assert args.num_eval_episodes == 5


# Each value is the argv after rl-garden-train-online in the named chapter.
CHAPTER_COMMANDS = {
    "01-preset": ["sac", "--config", "configs/online/sac_state.yaml", "--print-config"],
    "01-override": [
        "sac", "--config", "configs/online/sac_state.yaml", "--gamma", "0.95",
        "--print-config",
    ],
    "01-explain": [
        "sac", "--config", "configs/online/sac_state.yaml", "--gamma", "0.95",
        "--explain-param", "gamma",
    ],
    "02-sac": [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--total-timesteps", "2000", "--learning-starts", "100",
        "--training-freq", "4", "--batch-size", "64", "--buffer-size", "10000",
        "--hidden-dim", "64", "--gamma", "0.99", "--eval-freq", "0",
        "--log-freq", "500", "--no-capture-video", "--exp-name", "chapter02-sac",
    ],
    "02-ppo": [
        "ppo", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--num-steps", "32",
        "--num-minibatches", "2", "--update-epochs", "2",
        "--total-timesteps", "2048", "--gamma", "0.99", "--eval-freq", "0",
        "--no-capture-video", "--exp-name", "chapter02-ppo",
    ],
    "02-pushcube": [
        "sac", "--config", "configs/online/sac_state.yaml",
        "--env-id", "PushCube-v1", "--num-envs", "16", "--num-eval-envs", "1",
        "--total-timesteps", "2000", "--learning-starts", "100",
        "--training-freq", "4", "--buffer-size", "10000", "--batch-size", "64",
        "--hidden-dim", "64", "--buffer-device", "cuda", "--eval-freq", "0",
        "--log-freq", "500", "--no-capture-video", "--exp-name", "chapter02-pushcube",
    ],
    "02-pushcube-ppo": [
        "ppo", "--env-backend", "maniskill", "--env-id", "PushCube-v1",
        "--num-envs", "16", "--num-eval-envs", "1", "--num-steps", "32",
        "--num-minibatches", "2", "--update-epochs", "2",
        "--total-timesteps", "2048", "--eval-freq", "0",
        "--no-capture-video", "--exp-name", "chapter02-pushcube-ppo",
    ],
    "03-sac-rgb": [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera",
        "--encoder.backbone", "plain_conv", "--print-config",
    ],
    "03-ppo-rgb": [
        "ppo", "--config", "configs/online/ppo_rgb.yaml",
        "--obs.rgb", "base_camera", "--encoder.backbone", "plain_conv", "--print-config",
    ],
    "03-asymmetric": [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera",
        "--obs-groups.actor", "rgb_base_camera", "--obs-groups.critic",
        "rgb_base_camera", "state", "--encoder-sharing", "separate", "--print-config",
    ],
    "03-critic": [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera",
        "--encoder.backbone", "plain_conv", "--critic-encoder.backbone", "resnet18",
        "--encoder-sharing", "separate", "--print-config",
    ],
    "03-pretrained": [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera",
        "--encoder.backbone", "resnet10", "--encoder.pretrained-weights",
        "resnet10-imagenet", "--print-config",
    ],
    "06-save": [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "1000", "--batch-size", "32", "--hidden-dim", "32",
        "--learning-starts", "32", "--training-freq", "4", "--gamma", "0.99",
        "--eval-freq", "0", "--log-freq", "100", "--no-capture-video",
        "--total-timesteps", "200", "--save-replay-buffer", "--exp-name", "chapter06-save",
    ],
    "06-resume": [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "1000", "--batch-size", "32", "--hidden-dim", "32",
        "--learning-starts", "32", "--training-freq", "4", "--gamma", "0.99",
        "--eval-freq", "0", "--log-freq", "100", "--no-capture-video",
        "--total-timesteps", "400", "--save-replay-buffer",
        "--load-checkpoint", "runs/chapter06-save/checkpoints/final.pt",
        "--load-replay-buffer", "--exp-name", "chapter06-resume",
    ],
}
CHAPTER_EVALUATIONS = [
    ["--checkpoint-path", "runs/chapter02-sac/checkpoints/final.pt",
     "--algorithm", "auto", "--device", "cpu", "--num-eval-envs", "1",
     "--num-eval-episodes", "5"],
    ["--checkpoint-path", "runs/chapter06-resume/checkpoints/final.pt",
     "--algorithm", "auto", "--device", "cpu", "--num-eval-envs", "1",
     "--num-eval-episodes", "5"],
]


@pytest.mark.parametrize("name", CHAPTER_COMMANDS)
def test_chapter_print_config(name: str, tmp_path) -> None:
    argv = list(CHAPTER_COMMANDS[name])
    if "--load-checkpoint" in argv:
        # Preflight checks existence but does not deserialize checkpoint bytes.
        checkpoint = tmp_path / "final.pt"
        checkpoint.touch()
        argv[argv.index("--load-checkpoint") + 1] = str(checkpoint)
    if "--explain-param" in argv:
        index = argv.index("--explain-param")
        del argv[index:index + 2]
    if "--print-config" not in argv:
        argv.append("--print-config")
    result = run_entry("train_online", argv)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["selection"]["training_phase"] == "online"
    assert config["status"] == "preflight"


def test_chapter_explain_gamma() -> None:
    result = run_entry("train_online", CHAPTER_COMMANDS["01-explain"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["value"] == 0.95
    assert payload["source"]["kind"] == "CLI"


def test_chapter_unknown_preset_key(tmp_path) -> None:
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training.online import registry

    preset = tmp_path / "bad.yaml"
    preset.write_text("gamam: 0.95\n")
    with pytest.raises(ConfigError, match="gamam"):
        registry.parse_command(["sac", "--config", str(preset), "--print-config"])


@pytest.mark.parametrize("argv", CHAPTER_EVALUATIONS)
def test_chapter_evaluation_arguments(argv: list[str]) -> None:
    args = tyro.cli(EvalCheckpointArgs, args=argv)
    assert args.algorithm == "auto"
    assert args.num_eval_episodes == 5


# WP3: explicit argv is shared with the documented commands, phase by phase.
WP3_COMMANDS = {
    "04-bc": ("pretrain_offline", [
        "bc", "--offline-dataset", "runs/chapter04-data/pendulum.h5",
        "--dataset-backend", "h5", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--device", "cpu", "--buffer-device", "cpu", "--buffer-size", "256",
        "--batch-size", "16", "--action-low", "-2", "--action-high", "2",
        "--num-offline-steps", "20", "--eval-freq", "0", "--log-freq", "10",
        "--exp-name", "chapter04-bc",
    ]),
    "04-iql": ("pretrain_offline", [
        "iql", "--offline-dataset", "runs/chapter04-data/pendulum.h5",
        "--dataset-backend", "h5", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--device", "cpu", "--buffer-device", "cpu", "--buffer-size", "256",
        "--batch-size", "16", "--action-low", "-2", "--action-high", "2",
        "--n-critics", "2", "--critic-subsample-size", "2",
        "--num-offline-steps", "20", "--eval-freq", "0", "--log-freq", "10",
        "--exp-name", "chapter04-iql",
    ]),
    "04-calql": ("pretrain_offline", [
        "calql", "--offline-dataset", "runs/chapter04-data/pendulum.h5",
        "--buffer-device", "cpu", "--action-low", "-2", "--action-high", "2",
        "--print-config",
    ]),
    "05-wsrl-preset": ("train_off2on", [
        "wsrl", "--config", "configs/off2on/wsrl.yaml", "--print-config",
    ]),
    "05-calql-preset": ("train_off2on", [
        "calql", "--config", "configs/off2on/calql_halfcheetah_medium_v2_paper.yaml",
        "--print-config",
    ]),
    "05-offline": ("pretrain_offline", [
        "wsrl", "--offline-dataset", "runs/chapter04-data/pendulum.h5",
        "--dataset-backend", "h5", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--buffer-device", "cpu", "--buffer-size", "256",
        "--batch-size", "16", "--action-low", "-2", "--action-high", "2",
        "--hidden-dim", "32", "--actor-hidden-layers", "2", "--critic-hidden-layers", "2",
        "--n-critics", "2", "--critic-subsample-size", "2", "--gamma", "0.99",
        "--utd", "1", "--no-use-compile", "--num-offline-steps", "4",
        "--eval-freq", "0", "--log-freq", "2",
        "--exp-name", "chapter05-offline",
    ]),
    "05-online": ("train_off2on", [
        "wsrl", "--config", "configs/off2on/wsrl.yaml",
        "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "512", "--batch-size", "16", "--hidden-dim", "32",
        "--actor-hidden-layers", "2", "--critic-hidden-layers", "2",
        "--n-critics", "2", "--critic-subsample-size", "2", "--gamma", "0.99",
        "--utd", "1", "--training-freq", "4", "--no-use-compile",
        "--num-offline-steps", "0", "--num-online-steps", "64",
        "--warmup-steps", "16", "--learning-starts", "16", "--online-replay-mode", "empty",
        "--load-checkpoint", "runs/chapter05-offline/checkpoints/wsrl_offline_pretrained.pt",
        "--eval-freq", "0", "--log-freq", "16", "--no-capture-video",
        "--exp-name", "chapter05-online",
    ]),
    "07-gymnasium": ("train_online", [
        "sac", "--env-backend", "gymnasium", "--env-id", "TutorialLineReach-v0",
        "--gymnasium.import-modules", "ch07_my_env",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "512", "--batch-size", "16", "--hidden-dim", "32",
        "--learning-starts", "16", "--training-freq", "4", "--total-timesteps", "64",
        "--eval-freq", "0", "--log-freq", "32", "--no-capture-video",
        "--exp-name", "chapter07-gymnasium",
    ]),
    "07-custom": ("train_online", [
        "sac", "--env-backend", "custom", "--env-id", "PointReach-v0",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "512", "--batch-size", "16", "--hidden-dim", "32",
        "--learning-starts", "16", "--training-freq", "4", "--total-timesteps", "64",
        "--eval-freq", "0", "--log-freq", "32", "--no-capture-video",
        "--exp-name", "chapter07-custom",
    ]),
}
WP3_EVALUATION = [
    "--checkpoint-path", "runs/chapter04-iql/checkpoints/iql_offline_pretrained.pt",
    "--algorithm", "auto", "--device", "cpu", "--num-eval-envs", "1",
    "--num-eval-episodes", "2",
]


@pytest.mark.parametrize("name", WP3_COMMANDS)
def test_wp3_print_config(name: str, tmp_path) -> None:
    phase, documented = WP3_COMMANDS[name]
    argv = list(documented)
    for flag in ("--load-checkpoint", "--offline-dataset"):
        if flag in argv:
            path = tmp_path / ("model.pt" if flag == "--load-checkpoint" else "data.h5")
            path.touch()
            argv[argv.index(flag) + 1] = str(path)
    if "--print-config" not in argv:
        argv.append("--print-config")
    result = run_entry(phase, argv)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["status"] == "preflight"
    assert config["selection"]["algorithm"] == argv[0]


def test_wp3_evaluation_arguments() -> None:
    args = tyro.cli(EvalCheckpointArgs, args=WP3_EVALUATION)
    assert args.device == "cpu"
    assert args.num_eval_episodes == 2


# Non-phase commands have no --print-config action. Example tests execute the
# scripts; evaluation is parsed above; downstream help is checked below.
WP3_EXAMPLE_COMMANDS = {
    "04": ["python", "examples/tutorials/ch04_offline_bc.py", "--smoke",
           "--output-dir", "runs/chapter04-data"],
    "07": ["python", "examples/tutorials/ch07_my_env.py", "--smoke"],
}
WP3_PACKAGE_INSTALL = [
    "python", "-m", "pip", "install", "--no-deps", "--no-build-isolation",
    "-e", "tests/fixtures/downstream_extension",
]
WP3_DOWNSTREAM_COMMANDS = [
    ["--help"],
    ["aux_split_rppo", "--help"],
    ["aux_split_rppo", "--aux-weight", "0.2", "--print-config"],
]


def test_wp3_downstream_help(monkeypatch) -> None:
    import importlib
    import sys
    from pathlib import Path
    from rl_garden.training.online import registry

    # Match test_downstream_extension's isolated package import convention.
    registry.discover()
    monkeypatch.setattr(registry, "_entries", dict(registry._entries))
    root = Path(__file__).resolve().parents[1] / WP3_PACKAGE_INSTALL[-1]
    assert '[project.entry-points."rlgarden.algorithms"]' in (root / "pyproject.toml").read_text()
    monkeypatch.syspath_prepend(str(root))
    for name in list(sys.modules):
        if name.split(".")[0] == "rlg_downstream_example":
            monkeypatch.delitem(sys.modules, name)
    importlib.import_module("rlg_downstream_example.registration")
    listing = run_entry("train_online", WP3_DOWNSTREAM_COMMANDS[0])
    assert listing.returncode == 0
    assert "aux_split_rppo" in listing.stdout
    help_result = run_entry("train_online", WP3_DOWNSTREAM_COMMANDS[1])
    assert help_result.returncode == 0
    assert "--aux-weight" in help_result.stdout
    inspected = run_entry("train_online", WP3_DOWNSTREAM_COMMANDS[2])
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["inputs"]["aux_weight"] == 0.2


WP4_COMMANDS = {
    "08-ppo": ("train_online", [
        "ppo", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1", "--num-envs", "1",
        "--num-eval-envs", "1", "--total-timesteps", "200", "--eval-freq", "0", "--log-freq",
        "100", "--no-capture-video", "--num-steps", "8", "--num-minibatches", "2",
        "--update-epochs", "1", "--actor-factory",
        "ch08_transformer_ppo_actor:make_transformer_actor", "--exp-name", "chapter08-ppo",
    ]),
    "08-recurrent": ("train_online", [
        "recurrent_ppo", "--actor-factory", "ch08_transformer_ppo_actor:make_transformer_actor",
        "--print-config",
    ]),
    "08-sac": ("train_online", [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1", "--num-envs", "1",
        "--num-eval-envs", "1", "--total-timesteps", "200", "--eval-freq", "0", "--log-freq",
        "100", "--no-capture-video", "--buffer-device", "cpu", "--buffer-size", "1000",
        "--batch-size", "16", "--hidden-dim", "32", "--learning-starts", "16", "--training-freq",
        "4", "--actor-factory", "ch08_flow_sac_actor:make_flow_actor", "--exp-name",
        "chapter08-sac",
    ]),
    "08-calql-offline": ("pretrain_offline", [
        "calql", "--offline-dataset", "runs/chapter04-data/pendulum.h5", "--actor-factory",
        "ch08_flow_sac_actor:make_flow_actor", "--no-cql-importance-sample", "--print-config",
    ]),
    "08-calql-off2on": ("train_off2on", [
        "calql", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1", "--offline-dataset",
        "runs/chapter04-data/pendulum.h5", "--num-offline-steps", "4", "--actor-factory",
        "ch08_flow_sac_actor:make_flow_actor", "--no-cql-importance-sample", "--print-config",
    ]),
    "08-sac-extractor": ("train_online", [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera", "--actor-extractor",
        "ch08_backbone_extractor:BackboneExtractor", "--print-config",
    ]),
    "08-ppo-extractor": ("train_online", [
        "ppo", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera", "--actor-extractor",
        "ch08_backbone_extractor:BackboneExtractor", "--print-config",
    ]),
    "08-separate": ("train_online", [
        "sac", "--env-id", "PushCube-v1", "--obs.rgb", "base_camera", "--actor-extractor",
        "ch08_backbone_extractor:BackboneExtractor", "--critic-extractor",
        "ch08_backbone_extractor:BackboneExtractor", "--encoder-sharing", "separate",
        "--print-config",
    ]),
    "08-deterministic": ("train_online", [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1", "--actor-factory",
        "ch08_td3_shaped_sac:make_deterministic_actor", "--ent-coef", "0", "--target-policy-noise",
        "0.2", "--target-noise-clip", "0.5", "--actor-q-agg", "first", "--print-config",
    ]),
}

WP4_EVALUATIONS = [
    ["--checkpoint-path", "runs/chapter08-ppo/checkpoints/final.pt", "--device", "cpu",
     "--num-eval-envs", "1", "--num-eval-episodes", "2"],
    ["--checkpoint-path", "runs/chapter08-sac/checkpoints/final.pt", "--device", "cpu",
     "--num-eval-envs", "1", "--num-eval-episodes", "2"],
]


@pytest.mark.parametrize("name", WP4_COMMANDS)
def test_wp4_print_config(name, tmp_path, monkeypatch) -> None:
    from pathlib import Path
    from rl_garden.common.effective_config import resolve_import_path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples/tutorials"))
    phase, command = WP4_COMMANDS[name]
    argv = list(command)
    if "--offline-dataset" in argv:
        dataset = tmp_path / "data.h5"
        dataset.touch()
        argv[argv.index("--offline-dataset") + 1] = str(dataset)
    if "--print-config" not in argv:
        argv.append("--print-config")
    result = run_entry(phase, argv)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["selection"]["algorithm"] == argv[0]
    for flag in ("--actor-factory", "--actor-extractor", "--critic-extractor"):
        if flag in argv:
            path = argv[argv.index(flag) + 1]
            assert payload["inputs"][flag[2:].replace("-", "_")] == path
            assert callable(resolve_import_path(path))


@pytest.mark.parametrize("argv", WP4_EVALUATIONS)
def test_wp4_evaluation_arguments(argv) -> None:
    args = tyro.cli(EvalCheckpointArgs, args=argv)
    assert args.device == "cpu"
    assert args.num_eval_episodes == 2


FAMILY_COMMANDS = {
    "ppo-family": ("train_online", [
        "ppo", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--print-config",
    ]),
    "sac-family": ("train_online", [
        "sac", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--print-config",
    ]),
    "offline-rl": ("pretrain_offline", [
        "iql", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--offline-dataset", "demos/example.h5", "--print-config",
    ]),
    "offline-to-online": ("train_off2on", [
        "wsrl", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--offline-dataset", "demos/example.h5", "--print-config",
    ]),
    "imitation-and-generative": ("pretrain_offline", [
        "bc", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--offline-dataset", "demos/example.h5", "--print-config",
    ]),
    "model-based": ("train_online", [
        "tdmpc2", "--env-backend", "gymnasium", "--env-id", "Pendulum-v1",
        "--num-envs", "1", "--num-eval-envs", "1", "--print-config",
    ]),
}


@pytest.mark.parametrize("family", FAMILY_COMMANDS)
def test_family_print_config(family) -> None:
    entry, argv = FAMILY_COMMANDS[family]
    result = run_entry(entry, argv)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["selection"]["algorithm"] == argv[0]
    assert config["inputs"]["env_backend"] == "gymnasium"


BACKEND_COMMANDS = {
    "maniskill": [
        "sac", "--env-backend", "maniskill", "--env-id", "PushCube-v1",
        "--print-config",
    ],
    "mujoco": [
        "sac", "--env-backend", "mujoco", "--env-id", "InvertedPendulum-v4",
        "--print-config",
    ],
    "mujoco_warp": [
        "sac", "--env-backend", "mujoco_warp",
        "--env-id", "RlGarden-InvertedPendulum-Warp-v0", "--print-config",
    ],
    "minari": [
        "sac", "--env-backend", "minari", "--env-id", "metaworld/reach/expert-v0",
        "--minari.no-download", "--num-envs", "1", "--num-eval-envs", "1",
        "--buffer-device", "cpu", "--buffer-size", "512", "--batch-size", "16",
        "--hidden-dim", "32", "--learning-starts", "16", "--training-freq", "4",
        "--total-timesteps", "64", "--eval-freq", "0", "--log-freq", "32",
        "--no-capture-video", "--exp-name", "backend-minari", "--print-config",
    ],
    "custom": [
        "sac", "--env-backend", "custom", "--env-id", "PointReach-v0",
        "--num-envs", "1", "--num-eval-envs", "1", "--buffer-device", "cpu",
        "--buffer-size", "512", "--batch-size", "16", "--hidden-dim", "32",
        "--learning-starts", "16", "--training-freq", "4", "--total-timesteps", "64",
        "--eval-freq", "0", "--log-freq", "32", "--no-capture-video",
        "--exp-name", "backend-custom",
    ],
}


@pytest.mark.parametrize("backend", BACKEND_COMMANDS)
def test_backend_print_config(backend: str) -> None:
    argv = list(BACKEND_COMMANDS[backend])
    if "--print-config" not in argv:
        argv.append("--print-config")
    result = run_entry("train_online", argv)
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["status"] == "preflight"
    assert config["active_environment"]["backend"] == backend
    assert config["inputs"]["env_id"] == argv[argv.index("--env-id") + 1]
    if backend == "minari":
        assert config["active_environment"]["config"]["download"] is False
