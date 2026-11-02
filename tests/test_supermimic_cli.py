"""CLI tests for the online ``supermimic`` entrypoint: registry discovery,
Args defaults (paper Table A.5 and the constructor's), ``--print-config`` /
``--dry-run`` on the torch point-mass fake env (through the gymnasium
backend's ``vector_env_factory``), analytic-head preflight errors and a short
end-to-end run."""
from __future__ import annotations

import dataclasses
import inspect
import io
import json
from contextlib import redirect_stdout

import pytest
from _supermimic_fakes import PointMassTrackingEnv

ENV_FACTORY = "test_supermimic_cli:make_point_mass_env"
SMALL_ARGV = [
    "--num_envs", "4",
    "--num_eval_envs", "2",
    "--buffer_size", "512",
    "--buffer_device", "cpu",
    "--wm_hidden_dims", "16", "16",
    "--head_hidden_dims", "8",
    "--goal_encoder_dims", "16",
    "--decoder_dims", "16",
    "--critic_dims", "8",
    "--horizon", "3",
    "--wm_window", "2",
    "--wm_batches", "1",
    "--wm_batch_size", "8",
    "--imagination_batch", "8",
    "--warmup_updates", "1",
    "--sim_steps_per_iteration", "2",
    "--updates_per_iteration", "1",
    "--eval_freq", "0",
    "--log_type", "none",
    "--no-capture-video",
]


def make_point_mass_env(num_envs, seed, device, **env_kwargs):
    return PointMassTrackingEnv(num_envs, device=device, seed=seed, **env_kwargs)


def _env_argv(*, extra_state=("sim", "ctx", "goal", "time"), env_kwargs=None) -> list[str]:
    argv = [
        "--env_backend", "gymnasium",
        "--env_id", "PointMassTracking",
        "--gymnasium.vector_env_factory", ENV_FACTORY,
        "--gymnasium.vector_env_returns_torch",
        "--gymnasium.device", "cpu",
    ]
    if env_kwargs is not None:
        argv += ["--gymnasium.env_kwargs_json", json.dumps(env_kwargs)]
    if extra_state:
        argv += ["--obs.extra_state", *extra_state]
    return argv


@pytest.fixture
def online_registry(monkeypatch):
    from rl_garden.training.online import registry

    registry.discover()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return registry


def _dry_run(registry, *argv: str) -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        registry.run_cli(["supermimic", *argv, "--dry-run"])
    return json.loads(buffer.getvalue())


def test_registry_discovers_supermimic():
    from rl_garden.algorithms.supermimic import SuperMimic
    from rl_garden.training.online import registry
    from rl_garden.training.online.entrypoints.supermimic import build_supermimic

    registry.discover()
    entry = registry.entries()["supermimic"]

    assert entry.algorithm_cls() is SuperMimic
    assert entry.build_agent is build_supermimic


def test_args_defaults_match_paper_table_a5(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(["supermimic", "--log_type", "none", "--print-config"])
    config = json.loads(capsys.readouterr().out)

    assert config["selection"] == {"training_phase": "online", "algorithm": "supermimic"}
    inputs = config["inputs"]
    expected = {
        "horizon": 32,
        "wm_window": 8,
        "wm_batches": 10,
        "wm_batch_size": 2048,
        "imagination_batch": 4096,
        "gamma": 0.99,
        "lam": 0.95,
        "lr": 2e-4,
        "lr_final_ratio": 0.03,
        "lr_decay_updates": 250_000,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "action_noise_std": 0.1,
        "heads": "learned",
        "learning_starts": None,
        "amp_dtype": None,
        "actor_betas": [0.9, 0.95],
    }
    for name, value in expected.items():
        assert inputs[name] == value, name
    assert inputs["obs"]["extra_state"] == []


def test_args_defaults_equal_constructor_defaults():
    from rl_garden.algorithms.supermimic import SuperMimic
    from rl_garden.training.online.entrypoints.supermimic import SuperMimicArgs

    args = SuperMimicArgs()
    shared = [
        name
        for name in inspect.signature(SuperMimic.__init__).parameters
        if name in {field.name for field in dataclasses.fields(args)}
    ]
    assert {"heads", "horizon", "wm_hidden_dims", "buffer_size", "amp_dtype"} <= set(shared)
    for name in shared:
        constructor_default = inspect.signature(SuperMimic.__init__).parameters[name].default
        if name in {"seed", "logger", "std_log", "log_freq", "eval_freq", "num_eval_steps",
                    "checkpoint_dir", "checkpoint_freq", "save_replay_buffer", "save_final_checkpoint"}:
            continue  # owned by the shared run/logging/checkpoint Args
        assert getattr(args, name) == constructor_default, name


def test_print_config_accepts_privileged_observation_keys(capsys):
    from rl_garden.training.online import registry

    registry.run_cli(
        ["supermimic", "--log_type", "none", "--obs.extra_state", "sim", "ctx", "goal", "time", "--print-config"]
    )
    config = json.loads(capsys.readouterr().out)
    assert config["inputs"]["obs"]["extra_state"] == ["sim", "ctx", "goal", "time"]


@pytest.mark.parametrize("heads", ["learned", "analytic"])
def test_dry_run_builds_supermimic_on_point_mass(online_registry, heads):
    config = _dry_run(online_registry, *_env_argv(), *SMALL_ARGV, "--heads", heads)

    algorithm = config["algorithm"]
    assert algorithm["target"] == "rl_garden.algorithms.supermimic.SuperMimic"
    kwargs = algorithm["constructor_kwargs"]
    assert kwargs["heads"] == heads
    assert kwargs["horizon"] == 3
    assert kwargs["wm_hidden_dims"] == [16, 16]
    for unused in ("encoder_config", "obs_groups", "critic_encoder_config", "encoder_sharing"):
        assert unused not in kwargs
    observation_keys = set(config["runtime"]["train_observation_space"]["spaces"])
    assert observation_keys == {"state", "state_sim", "state_ctx", "state_goal", "state_time"}


def test_analytic_heads_without_analytic_mdp_raise(online_registry):
    argv = [*_env_argv(extra_state=(), env_kwargs={"extra_keys": False}), *SMALL_ARGV, "--heads", "analytic"]
    with pytest.raises(ValueError, match="analytic_mdp"):
        _dry_run(online_registry, *argv)


def test_analytic_heads_without_state_time_raise(online_registry):
    argv = [
        *_env_argv(extra_state=("sim", "ctx", "goal"), env_kwargs={"time_key": False}),
        *SMALL_ARGV,
        "--heads", "analytic",
    ]
    with pytest.raises(ValueError, match="state_time"):
        _dry_run(online_registry, *argv)


def test_cli_trains_and_checkpoints(online_registry, tmp_path):
    from rl_garden.common.checkpoint import load_checkpoint_file

    online_registry.run_cli(
        [
            "supermimic",
            *_env_argv(),
            *SMALL_ARGV,
            "--heads", "analytic",
            "--learning_starts", "16",
            "--total_timesteps", "64",
            "--log_freq", "16",
            "--log_dir", str(tmp_path),
            "--exp_name", "run",
        ]
    )

    metadata = load_checkpoint_file(tmp_path / "run" / "checkpoints" / "final.pt", map_location="cpu")["metadata"]
    assert metadata["algorithm_class"] == "SuperMimic"
    assert metadata["hyperparameters"]["heads"] == "analytic"
    assert metadata["hyperparameters"]["sim_steps_per_iteration"] == 2
    assert metadata["hyperparameters"]["horizon"] == 3
