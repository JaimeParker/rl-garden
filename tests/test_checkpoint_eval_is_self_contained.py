"""Evaluating a checkpoint needs the checkpoint and an evaluation env only.

Every registered, evaluable algorithm trains a few real steps on CPU with the
inputs ``test_entrypoint_builds`` gives it (plus the optional datasets and
checkpoints it can read); then every input file is deleted and a copy of the
checkpoint alone in a directory is evaluated with the training env, the dataset
readers and every other ``torch.load`` disabled. The rebuilt agent must act as
the trained one does, and leave its env alone until evaluation starts.
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import math
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pytest
import test_entrypoint_builds as builds
import torch
from _ncrl_online_synthetic import ENV_ACTION_DIM, make_pretrain_artifacts
from gymnasium import spaces

from rl_garden.common.checkpoint import without_local_paths
from rl_garden.common.effective_config import json_value, resolve_active_environment
from rl_garden.common.env_args import make_env_request
from rl_garden.common.utils import get_device
from rl_garden.envs import backend_registry
from rl_garden.training import evaluation, inspection
from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry
from rl_garden.training.evaluation import EvalCheckpointArgs, evaluate_checkpoint

shared_inputs = builds.shared_inputs
cpu_only_and_restored_rng = builds.cpu_only_and_restored_rng

NCRL_ENV_ID = "RlgSelfContainedEvalNCRL-v0"


class _NCRLEnv(builds._SmallObservationEnv):
    """The image-only env of the NCRL synthetic dataset's walker embodiment."""

    def __init__(self) -> None:
        super().__init__(state=False, camera="default", image_size=builds.IMAGE_ONLY_SIZE)
        self.action_space = spaces.Box(-1.0, 1.0, (ENV_ACTION_DIM,), np.float32)


if NCRL_ENV_ID not in gym.registry:
    gym.register(NCRL_ENV_ID, entry_point=_NCRLEnv)

#: 8-step episodes, so an evaluation takes one.
STATE_ENV_ID = builds.LOCOMOTION_NAMED_PENDULUM_ID
SMALL_TRAINING = {
    "env_id": STATE_ENV_ID,
    "total_timesteps": "32",
    "num_offline_steps": "2",
    "num_online_steps": "16",
    "num_steps": "8",
    "training_freq": "4",
    "utd": "0.5",
    "num_eval_steps": "8",
    "update_epochs": "1",
    "horizon_steps": "4",
    # Model-based algorithms.
    "episode_length": str(builds.EPISODE_LENGTH),
    "seed_steps": "8",
    "num_samples": "8",
    "num_elites": "4",
    "num_pi_trajs": "2",
    "iterations": "1",
    "latent_dim": "16",
    "mlp_dim": "16",
    "train_ratio": "2",
    "imag_horizon": "2",
    # Sequence and chunk replay.
    "burn_in_len": "2",
    "learning_len": "2",
    "forward_len": "1",
    "horizon_length": "2",
    "vae_iterations": "2",
}

# (phase, algorithm) -> training flags on top of SMALL_TRAINING and the build
# test's OVERRIDES.
TRAINING_OVERRIDES: dict[tuple[str, str], dict[str, str | bool]] = {
    ("online", "ncrl"): {
        "env_id": NCRL_ENV_ID, "gymnasium.import_modules": __name__, "total_timesteps": "48",
        "backbone": "dreamer_v3", "online_ratio": "0.5", "num_retrieve_episodes": "3",
        "bc_train_steps": "2", "bc_batch_size": "2", "batch_size": "2"},
    ("online", "dreamer_v3"): {"learning_starts": "16", "batch_size": "2"},
    ("online", "dagger"): {
        "rollout_steps_per_round": "8", "gradient_steps_per_round": "2", "batch_size": "8"},
    **dict.fromkeys(
        (("online", "recurrent_sac"), ("online", "transformer_sac")),
        {"total_timesteps": "64", "learning_starts": "48", "memory_len": "2"},
    ),
    ("offline", "bppo"): {"critic_warmup_steps": "1", "num_offline_steps": "2", "eval_freq": "1"},
    **dict.fromkeys(
        (("offline", "unio4"), ("offline", "unio4_ope")),
        {"critic_warmup_steps": "1", "bc_ensemble_steps": "1", "num_offline_steps": "4",
         "eval_freq": "1"},
    ),
}

_PRIOR_DATA = {"offline_buffer_size": "96", "action_low": "-2", "action_high": "2"}

# (phase, algorithm) -> inputs it can train with but does not require.
OPTIONAL_INPUTS = {
    ("online", "rlpd"): lambda s: {"offline_dataset": str(s.h5), **_PRIOR_DATA},
    ("online", "rlpd_hybrid"): lambda s: {"offline_dataset": str(s.h5), **_PRIOR_DATA},
    ("online", "explore"): lambda s: {"offline_dataset": str(s.h5), **_PRIOR_DATA},
    ("online", "acrlpd"): lambda s: {"offline_dataset": str(s.h5), **_PRIOR_DATA},
    ("online", "supe"): lambda s: {"offline_dataset": str(s.h5), **_PRIOR_DATA},
    ("online", "dppo"): lambda s: {"bc_checkpoint": str(s.diffusion_bc_checkpoint)},
    ("online", "diffusion_cm_distill_online"): lambda s: {
        "bc_checkpoint": str(s.diffusion_bc_checkpoint)},
}

#: Algorithms that also train with an mmap replay buffer, deleted afterwards.
MMAP_REPLAY = {
    ("online", name) for name in ("sac", "rlpd", "rlpd_hybrid", "explore", "supe", "td3", "drqv2")
}

FAST_LANE = {("online", name) for name in ("rlpd", "dppo", "supe")}

# (phase, algorithm) -> why it cannot train here; a built, untrained agent's
# checkpoint is evaluated instead.
UNTRAINED: dict[tuple[str, str], str] = {
    ("online", "gail"): "demonstrations come only from a D4RL dataset (d4rl_legacy, mujoco_py)",
}

# (phase, algorithm) -> why its checkpoint cannot be evaluated here.
EXEMPTIONS: dict[tuple[str, str], str] = {}


def _evaluable(phase: str, name: str) -> bool:
    factory = builds._registered(phase)[name].algorithm_cls
    return factory is None or factory().checkpoint_eval_unsupported_reason is None


EVALUABLE = {key for key in builds.REGISTERED if _evaluable(*key)}
CASES = sorted(EVALUABLE - EXEMPTIONS.keys())


def _copy_input_files(inputs: dict[str, str | bool], directory: Path) -> dict[str, str | bool]:
    copied = {}
    directory.mkdir()
    for name, value in inputs.items():
        source = Path(value) if isinstance(value, str) else None
        if source is not None and source.is_absolute() and source.exists():
            target = directory / name / source.name
            target.parent.mkdir()
            (shutil.copytree if source.is_dir() else shutil.copy)(source, target)
            value = str(target)
        copied[name] = value
    return copied


@contextlib.contextmanager
def _captured_agents(monkeypatch):
    """Every agent a builder constructs in this scope."""
    agents = []
    construct = inspection.construct_agent

    def construct_and_keep(target, /, **kwargs):
        agents.append(construct(target, **kwargs))
        return agents[-1]

    with monkeypatch.context() as patch:
        patch.setattr(inspection, "construct_agent", construct_and_keep)
        yield agents


def _train(phase: str, name: str, shared, tmp_path: Path, monkeypatch) -> tuple[Path, Path, object]:
    """``(checkpoint, input directory, trained agent)``."""
    key = (phase, name)
    input_dir = tmp_path / "inputs"
    inputs = builds.inputs_for(phase, name, shared, tmp_path)
    inputs.update(OPTIONAL_INPUTS.get(key, lambda s: {})(shared))
    if key == ("online", "ncrl"):
        artifacts = make_pretrain_artifacts(tmp_path / "ncrl")
        inputs.update(
            pretrained_world_model=str(artifacts["checkpoint"]),
            retrieval_db=str(artifacts["retrieval_db"]),
            prior_dataset_dir=str(artifacts["dataset_dir"]),
        )
    inputs = _copy_input_files(inputs, input_dir)
    if key == ("online", "ncrl"):
        shutil.rmtree(tmp_path / "ncrl")
    if key in MMAP_REPLAY:
        inputs["mmap_dir"] = str(input_dir / "mmap")
    values = {
        **SMALL_TRAINING, **builds.OVERRIDES.get(key, {}),
        **TRAINING_OVERRIDES.get(key, {}), **inputs,
    }
    argv = builds.build_argv(
        phase, name, tmp_path / "runs", with_eval=key in builds.EVAL_ONLY, inputs=values
    )
    with _captured_agents(monkeypatch) as agents:
        if key in UNTRAINED:
            checkpoint = _save_untrained(phase, name, argv, tmp_path / "runs")
        else:
            result = builds.run_entry(builds.PHASES[phase][0], argv)
            assert result.returncode == 0, f"{result.stderr[-1500:]!r} {result.stdout[-500:]!r}"
            checkpoints = sorted((tmp_path / "runs").glob("*/checkpoints/*.pt"))
            assert checkpoints, f"{key} wrote no checkpoint"
            # off2on also writes the offline phase's checkpoint.
            checkpoint = next((c for c in checkpoints if c.name == "final.pt"), checkpoints[-1])
    return checkpoint, input_dir, agents[-1]


def _save_untrained(phase: str, name: str, argv: list[str], runs: Path) -> Path:
    registry = importlib.import_module(builds.PHASES[phase][1]).registry
    args = registry.parse_args(argv)
    env = backend_registry.make_evaluation_env(args.env_backend, make_env_request(args))
    try:
        if name == "gail":
            args.demo_env_id = None
        agent = builds._registered(phase)[name].build_agent(args, env, env, None, None)
        if name == "gail":
            args.demo_env_id = agent.demo_env_id = "halfcheetah-expert-v2"
        agent.run_config = {
            "inputs": json_value(args),
            "active_environment": json_value(resolve_active_environment(args)),
        }
        return agent.save(runs / name / "checkpoints" / "final.pt")
    finally:
        env.close()


def _alone(checkpoint: Path, tmp_path: Path) -> Path:
    lone = tmp_path / "lone" / "checkpoint_dir" / checkpoint.name
    lone.parent.mkdir(parents=True)
    return Path(shutil.copy(checkpoint, lone))


def _forbid(monkeypatch, owner, attribute: str) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError(f"evaluation called {attribute}")

    monkeypatch.setattr(owner, attribute, forbidden)


class _EnvGuard:
    """The evaluation env's ``reset``/``step`` are usable only once
    ``evaluate_agent`` runs, and it is closed once."""

    def __init__(self, patch) -> None:
        self.evaluating = False
        self.closes = 0
        self.agent = None
        make_env, evaluate_agent = evaluation.make_evaluation_env, evaluation.evaluate_agent

        def make_guarded_env(backend, request):
            env = make_env(backend, request)
            for name in ("reset", "step"):
                setattr(env, name, self._guarded(name, getattr(env, name)))
            close = env.close

            def counted_close():
                self.closes += 1
                close()

            env.close = counted_close
            return env

        def evaluate_guarded(agent, *args, **kwargs):
            self.evaluating = True
            self.agent = agent
            return evaluate_agent(agent, *args, **kwargs)

        patch.setattr(evaluation, "make_evaluation_env", make_guarded_env)
        patch.setattr(evaluation, "evaluate_agent", evaluate_guarded)

    def _guarded(self, name, method):
        def call(*args, **kwargs):
            assert self.evaluating, f"env.{name} was called while the agent was being built"
            return method(*args, **kwargs)

        return call


def _evaluate(checkpoint: Path, phase: str, name: str, monkeypatch, **kwargs):
    """``(result, evaluated agent)``, with everything but the checkpoint and
    the eval env disabled."""
    backend = backend_registry._get_backend("gymnasium")
    with monkeypatch.context() as patch:
        guard = _EnvGuard(patch)
        _forbid(patch, backend, "make_train_env")
        _forbid(patch, h5py, "File")
        _forbid(patch, np, "load")
        load = torch.load

        def load_only_the_checkpoint(path, *args, **kwargs):
            assert Path(path) == checkpoint, f"evaluation read {path}"
            return load(path, *args, **kwargs)

        patch.setattr(torch, "load", load_only_the_checkpoint)
        result = evaluate_checkpoint(
            EvalCheckpointArgs(
                checkpoint_path=str(checkpoint), phase=phase, algorithm=name, num_eval_envs=1,
                # Not the training seed: the networks must come from the checkpoint.
                num_eval_episodes=1, max_eval_steps=50, seed=12345, device="cpu", **kwargs,
            )
        )
    assert guard.closes == 1
    return result, guard.agent


def _eval_actions(agent, steps: int = 3) -> list[torch.Tensor]:
    """The evaluation actions of ``agent`` for fixed observations and noise."""
    generator = torch.Generator().manual_seed(0)
    agent.policy.eval()
    agent._eval_start_hook()
    running = torch.zeros(1, dtype=torch.bool)
    actions = []
    for step in range(steps):
        observation = {
            key: torch.randint(0, 256, (1, *space.shape), dtype=torch.uint8, generator=generator)
            if space.dtype == np.uint8
            else torch.randn((1, *space.shape), generator=generator)
            for key, space in agent.env.single_observation_space.spaces.items()
        }
        torch.manual_seed(step)
        with torch.no_grad():
            action, critic_action = agent._eval_action_and_critic_action(observation)
        agent._eval_step_hook(
            observation, critic_action, torch.zeros(1), running, running, {}
        )
        actions.append(action)
    return actions


def _assert_acts_alike(evaluated, trained) -> None:
    # Evaluated with one env, trained with several.
    trained.eval_env = evaluated.eval_env
    for step, (actual, expected) in enumerate(
        zip(_eval_actions(evaluated), _eval_actions(trained), strict=True)
    ):
        assert torch.allclose(actual, expected, atol=1e-6), (step, actual, expected)


def test_tables_agree_with_the_registries():
    for table in (
        TRAINING_OVERRIDES, OPTIONAL_INPUTS, MMAP_REPLAY, FAST_LANE, UNTRAINED, EXEMPTIONS
    ):
        assert set(table) <= EVALUABLE
    assert all(UNTRAINED.values()) and all(EXEMPTIONS.values())
    assert len(CASES) >= 60


@pytest.mark.parametrize(
    "phase,name",
    [
        pytest.param(*key, id="-".join(key), marks=() if key in FAST_LANE else pytest.mark.slow)
        for key in CASES
    ],
)
def test_evaluation_needs_only_the_checkpoint(phase, name, tmp_path, shared_inputs, monkeypatch):
    checkpoint, input_dir, trained = _train(phase, name, shared_inputs, tmp_path, monkeypatch)
    shutil.rmtree(input_dir)
    checkpoint = _alone(checkpoint, tmp_path)

    result, evaluated = _evaluate(checkpoint, phase, name, monkeypatch)

    metrics = result["metrics"]
    assert metrics["episodes_completed"] == 1.0
    assert all(math.isfinite(value) for value in metrics.values()), metrics
    _assert_acts_alike(evaluated, trained)
    assert not input_dir.exists()


def _edit_checkpoint(checkpoint: Path, edit) -> None:
    contents = torch.load(checkpoint)
    edit(contents)
    torch.save(contents, checkpoint)


def _forget_run_config(contents) -> None:
    del contents["metadata"]["run_config"]


def _eval_args(checkpoint: Path, **kwargs) -> EvalCheckpointArgs:
    return EvalCheckpointArgs(
        checkpoint_path=str(checkpoint), num_eval_envs=1, num_eval_episodes=1,
        max_eval_steps=50, device="cpu", **kwargs,
    )


@pytest.mark.parametrize(
    "phase,name",
    [("online", "sac"), ("online", "ppo"), ("online", "dppo"), ("off2on", "iql"), ("offline", "iql")],
)
def test_checkpoint_without_a_run_config(
    phase, name, tmp_path, shared_inputs, monkeypatch, caplog
):
    checkpoint, _, trained = _train(phase, name, shared_inputs, tmp_path, monkeypatch)
    _edit_checkpoint(checkpoint, _forget_run_config)

    # config.json beside it, as the run left it.
    _, evaluated = _evaluate(checkpoint, phase, name, monkeypatch)
    _assert_acts_alike(evaluated, trained)

    # Nothing beside it: never the Args defaults.
    lone = _alone(checkpoint, tmp_path)
    with pytest.raises(SystemExit, match="embeds no run configuration.*--env-backend and --env-id"):
        evaluate_checkpoint(_eval_args(lone))
    with pytest.raises(SystemExit, match="--env-backend and --env-id"):
        evaluate_checkpoint(_eval_args(lone, env_id=STATE_ENV_ID))
    with pytest.raises(SystemExit, match="--config-path .*absent.json does not exist"):
        evaluate_checkpoint(_eval_args(lone, config_path=str(tmp_path / "absent.json")))

    # With the environment given, a setting the checkpoint does not record takes
    # its default, with a warning. The network widths are recorded (``net_arch``),
    # so the agent is rebuilt in the size it was trained in.
    flags = {"env_backend": "gymnasium", "env_id": STATE_ENV_ID, "phase": phase, "algorithm": name}
    evaluate_checkpoint(_eval_args(lone, **flags))
    assert "takes its default" in caplog.text

    # The run's config.json, named explicitly.
    config = checkpoint.parent.parent / "config.json"
    evaluate_checkpoint(_eval_args(lone, config_path=str(config), phase=phase, algorithm=name))


def test_run_config_is_config_json_without_its_paths(tmp_path, shared_inputs, monkeypatch):
    for phase, name in (("online", "sac"), ("off2on", "iql"), ("offline", "iql")):
        directory = tmp_path / phase
        directory.mkdir()
        checkpoint, _, _ = _train(phase, name, shared_inputs, directory, monkeypatch)
        contents = torch.load(checkpoint)
        config = json.loads((checkpoint.parent.parent / "config.json").read_text())
        assert contents["metadata"]["run_config"] == without_local_paths(config)
        assert config["selection"]["algorithm"] == name
        assert contents["format_version"] == 1


def test_embedded_run_config_of_an_unregistered_backend(tmp_path, shared_inputs, monkeypatch):
    checkpoint, _, _ = _train("online", "ppo", shared_inputs, tmp_path, monkeypatch)
    lone = _alone(checkpoint, tmp_path)

    def from_elsewhere(contents):
        run_config = contents["metadata"]["run_config"]
        run_config["inputs"]["env_backend"] = "robot_at_home"
        run_config["inputs"]["downstream_only"] = {"gain": 2}
        run_config["active_environment"] = {"backend": "robot_at_home", "config": {"port": 1}}

    _edit_checkpoint(lone, from_elsewhere)
    with pytest.raises(SystemExit, match="robot_at_home.*--env-backend"):
        evaluate_checkpoint(_eval_args(lone))
    evaluate_checkpoint(_eval_args(lone, env_backend="gymnasium"))


@dataclasses.dataclass
class _Args:
    dataset: str | None = None
    teacher: str = ""
    ratio: float = 0.5


class _Registry(BaseAlgorithmRegistry):
    package_name = "unused"
    phase_name = "online"


def test_training_only_inputs_become_absent_values():
    registry = _Registry()
    registry.register("names", _Args, lambda args: None, training_only_inputs=("dataset", "teacher"))
    assert registry.entries()["names"].training_only_inputs == {"dataset": None, "teacher": ""}

    registry = _Registry()
    registry.register("mapping", _Args, lambda args: None, training_only_inputs={"ratio": 1.0})
    assert registry.entries()["mapping"].training_only_inputs == {"ratio": 1.0}

    registry = _Registry()
    registry.register("none", _Args, lambda args: None)
    assert registry.entries()["none"].training_only_inputs == {}


@pytest.mark.parametrize(
    "inputs,match",
    [(("datset",), "not fields"), ({"datset": None}, "not fields"), (("ratio",), "mapping")],
)
def test_training_only_inputs_are_checked_at_registration(inputs, match):
    with pytest.raises(ValueError, match=match):
        _Registry().register("bad", _Args, lambda args: None, training_only_inputs=inputs)


class _TakesDevice:
    def __init__(self, device="auto", **kwargs):
        self.device = device


class _ForwardsDevice(_TakesDevice):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)


class _NoDevice:
    def __init__(self, scale=1.0):
        self.scale = scale


def test_construct_agent_uses_the_evaluation_device_only_inside_the_scope():
    evaluation = inspection.CheckpointEvaluation(Path("x.pt"), {}, "cpu:7")
    with inspection.checkpoint_evaluation(evaluation):
        assert inspection.current_evaluation() is evaluation
        assert inspection.construct_agent(_TakesDevice).device == "cpu:7"
        assert inspection.construct_agent(_TakesDevice, device="cuda").device == "cpu:7"
        assert inspection.construct_agent(_ForwardsDevice).device == "cpu:7"
        assert inspection.construct_agent(_NoDevice).scale == 1.0
    assert inspection.current_evaluation() is None
    assert inspection.construct_agent(_TakesDevice).device == "auto"
    with pytest.raises(RuntimeError), inspection.checkpoint_evaluation(evaluation):
        raise RuntimeError
    assert inspection.current_evaluation() is None


class _HidesDevice(_TakesDevice):
    """Neither takes ``device`` nor forwards keywords."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.device = get_device(self.device)


def test_auto_device_resolves_to_the_evaluation_device_inside_the_scope():
    evaluation_ = inspection.CheckpointEvaluation(Path("x.pt"), {}, "cpu:3")
    before = get_device("auto")
    with inspection.checkpoint_evaluation(evaluation_):
        assert get_device("auto") == torch.device("cpu:3")
        assert get_device("cuda:1") == torch.device("cuda:1")
        assert inspection.construct_agent(_HidesDevice).device == torch.device("cpu:3")
    assert get_device("auto") == before


def test_every_evaluable_algorithm_takes_the_evaluation_device():
    for phase, name in sorted(EVALUABLE):
        factory = builds._registered(phase)[name].algorithm_cls
        if factory is not None:
            assert inspection._accepts_device(factory()), (phase, name)


CUDA_IS_AVAILABLE = torch.cuda.is_available
CUDA_AVAILABLE = CUDA_IS_AVAILABLE()
DEVICE_CASES = [
    ("online", "ppo"), ("online", "recurrent_ppo"), ("online", "transformer_ppo"),
    ("online", "sac"), ("online", "recurrent_sac"), ("online", "supe"),
    ("online", "dppo"), ("online", "tdmpc2"), ("online", "dreamer_v3"),
    ("offline", "iql"), ("off2on", "iql"), ("off2on", "wsrl"),
]


def _module_devices(agent) -> set[str]:
    modules = [value for value in vars(agent).values() if isinstance(value, torch.nn.Module)]
    assert agent.policy in modules
    return {
        tensor.device.type
        for module in modules
        for tensor in (*module.parameters(), *module.buffers())
    }


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is unavailable")
@pytest.mark.parametrize("phase,name", DEVICE_CASES, ids=["-".join(key) for key in DEVICE_CASES])
def test_agent_runs_on_the_evaluation_device(phase, name, tmp_path, shared_inputs, monkeypatch):
    checkpoint, _, _ = _train(phase, name, shared_inputs, tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", CUDA_IS_AVAILABLE)
    for agent_device, env_device in (("cpu", "cpu"), ("cuda", "cpu"), ("cpu", "cuda:0"), ("cuda", "cuda:0")):
        args = _eval_args(checkpoint, phase=phase, algorithm=name)
        args.device = agent_device
        args.gymnasium.device = env_device
        with _captured_agents(monkeypatch) as agents:
            metrics = evaluate_checkpoint(args)["metrics"]
        assert all(math.isfinite(value) for value in metrics.values())
        assert _module_devices(agents[-1]) == {agent_device}
        assert agents[-1].device.type == agent_device


@pytest.mark.parametrize(
    "name,extra_key,argument",
    [("supe", "opal_vae", "opal_checkpoint")],
)
def test_a_checkpoint_without_the_embedded_network_reads_the_original_file(
    name, extra_key, argument, tmp_path, shared_inputs, monkeypatch
):
    checkpoint, input_dir, trained = _train("online", name, shared_inputs, tmp_path, monkeypatch)
    _edit_checkpoint(checkpoint, lambda contents: contents["state"]["extra"].pop(extra_key))
    with _captured_agents(monkeypatch) as agents:
        evaluate_checkpoint(_eval_args(checkpoint))
    _assert_acts_alike(agents[-1], trained)

    # Moved: its path comes from the eval CLI, else evaluation fails naming it.
    (source,) = (input_dir / argument).iterdir()
    moved = shutil.move(source, tmp_path / "moved.pt")
    with pytest.raises(FileNotFoundError, match=source.name):
        evaluate_checkpoint(_eval_args(checkpoint))
    with _captured_agents(monkeypatch) as agents:
        evaluate_checkpoint(_eval_args(checkpoint, **{argument: str(moved)}))
    _assert_acts_alike(agents[-1], trained)


@pytest.mark.parametrize("name,argument", [("supe", "opal_checkpoint")])
def test_resume_restores_the_embedded_network(
    name, argument, tmp_path, shared_inputs, monkeypatch, caplog
):
    checkpoint, input_dir, _ = _train("online", name, shared_inputs, tmp_path, monkeypatch)
    key = ("online", name)
    other = input_dir / argument / "other.pt"
    (source,) = (input_dir / argument).iterdir()
    source.rename(other)
    inputs = builds.inputs_for("online", name, shared_inputs, tmp_path)
    inputs[argument] = str(input_dir / argument / "gone.pt")
    values = {**SMALL_TRAINING, **builds.OVERRIDES.get(key, {}), **inputs, "total_timesteps": "48"}
    argv = builds.build_argv("online", name, tmp_path / "resumed", inputs=values)

    with pytest.raises(FileNotFoundError, match="gone.pt"):
        builds.run_entry("train_online", argv)

    resume = [*argv, "--load-checkpoint", str(checkpoint)]
    result = builds.run_entry("train_online", resume)
    assert result.returncode == 0, result.stderr[-1500:]
    (resumed_checkpoint,) = (tmp_path / "resumed").glob("*/checkpoints/*.pt")
    extra_key = "opal_vae"
    weights_key = "state_dict"
    before = torch.load(checkpoint)["state"]["extra"][extra_key][weights_key]
    after = torch.load(resumed_checkpoint)["state"]["extra"][extra_key][weights_key]
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)

    # The flag is not needed either, unless the checkpoint embeds no network.
    flag = f"--{argument.replace('_', '-')}"
    without_flag = [*resume[: resume.index(flag)], *resume[resume.index(flag) + 2 :]]
    result = builds.run_entry("train_online", without_flag)
    assert result.returncode == 0, result.stderr[-1500:]
    legacy = tmp_path / "legacy.pt"
    shutil.copy(checkpoint, legacy)
    _edit_checkpoint(legacy, lambda contents: contents["state"]["extra"].pop(extra_key))
    without_flag[without_flag.index(str(checkpoint))] = str(legacy)
    with pytest.raises(FileNotFoundError, match="does not embed"):
        builds.run_entry("train_online", without_flag)

    # The same file is accepted silently, a differing one is reported.
    resume[resume.index(flag) + 1] = str(other)
    assert builds.run_entry("train_online", resume).returncode == 0
    assert "not the network embedded" not in caplog.text

    def perturb(contents):
        for value in contents["state"]["policy"].values():
            if value.is_floating_point():
                value.add_(1.0)

    _edit_checkpoint(other, perturb)
    assert builds.run_entry("train_online", resume).returncode == 0
    assert f"{flag} {other} is not the network embedded" in caplog.text


def test_acrlpd_checkpoint_of_a_critic_only_run_evaluates_and_resumes_without_the_phase(
    tmp_path, shared_inputs, monkeypatch
):
    checkpoint, _, trained = _train("online", "acrlpd", shared_inputs, tmp_path, monkeypatch)
    phase = {
        "duration_steps": 30, "update_actor": False, "update_critic": True,
        "update_encoder": False, "random_action_prob": 0.0,
    }

    def record_phase(contents):
        inputs = contents["metadata"]["run_config"]["inputs"]
        assert inputs["critic_only_steps"] == 0
        inputs["critic_only_steps"] = phase["duration_steps"]
        contents["metadata"]["hyperparameters"]["initial_training_phase"] = phase
        contents["state"]["training_state"]["initial_phase_start_step"] = 0

    _edit_checkpoint(checkpoint, record_phase)
    config = checkpoint.parent.parent / "config.json"
    (tmp_path / "config.json").write_text(
        config.read_text().replace('"critic_only_steps": 0', '"critic_only_steps": 30')
    )
    assert '"critic_only_steps": 30' in (tmp_path / "config.json").read_text()
    for config_path in (None, str(tmp_path / "config.json")):
        with _captured_agents(monkeypatch) as agents:
            evaluate_checkpoint(_eval_args(checkpoint, config_path=config_path))
        _assert_acts_alike(agents[-1], trained)

    values = {
        **SMALL_TRAINING, **builds.OVERRIDES.get(("online", "acrlpd"), {}),
        "total_timesteps": "48", "load_checkpoint": str(checkpoint),
    }
    argv = builds.build_argv("online", "acrlpd", tmp_path / "resumed", inputs=values)
    with pytest.raises(ValueError, match="critic-only-steps"):
        builds.run_entry("train_online", [*argv, "--critic-only-steps", "30"])
    with _captured_agents(monkeypatch) as agents:
        assert builds.run_entry("train_online", argv).returncode == 0
    assert agents[-1].initial_training_phase is None
    assert agents[-1]._global_step == 48


def test_evaluation_allocates_no_replay_storage(tmp_path, shared_inputs, monkeypatch):
    checkpoint, _, _ = _train("online", "rlpd", shared_inputs, tmp_path, monkeypatch)
    _, agent = _evaluate(checkpoint, "online", "rlpd", monkeypatch)
    assert agent.replay_buffer.rewards.is_meta
    assert agent.replay_buffer.obs["state"].is_meta
    assert agent.env.unwrapped is agent.eval_env.unwrapped


def test_evaluation_does_not_read_pretrained_encoder_weights(tmp_path, monkeypatch):
    from rl_garden.encoders.resnet import ResNetEncoder
    from rl_garden.training.evaluation import _encoder_weights_from_checkpoint

    monkeypatch.setenv("RL_GARDEN_PRETRAINED_DIR", str(tmp_path))
    load_pretrained = ResNetEncoder.load_pretrained
    kwargs = {"pretrained_weights": "absent"}
    with pytest.raises(FileNotFoundError):
        ResNetEncoder(**_resnet_kwargs(), **kwargs)
    with _encoder_weights_from_checkpoint():
        ResNetEncoder(**_resnet_kwargs(), **kwargs)
    assert ResNetEncoder.load_pretrained is load_pretrained


def _resnet_kwargs():
    return {"observation_space": spaces.Box(0, 255, (3, 32, 32), np.uint8)}
