"""Every registered training entrypoint builds its env and agent (``--dry-run``).

CLI tests stop at ``--print-config`` and wrapper tests use fake envs, so an
entrypoint whose builder cannot construct its agent would otherwise pass the
fast lane. Each registered algorithm of the online, off2on and offline phases
runs the real ``--dry-run`` path on CPU against gymnasium's ``Pendulum-v1``
(or a tiny registered image env), with an H5 dataset written here.
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import random
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

try:
    from tests._cli import run_entry
except ModuleNotFoundError as exc:
    if exc.name != "tests._cli":
        raise
    # Some environments install a top-level tests package that shadows this one.
    from _cli import run_entry

PHASES = {
    "online": ("train_online", "rl_garden.training.online.registry"),
    "off2on": ("train_off2on", "rl_garden.training.off2on.registry"),
    "offline": ("pretrain_offline", "rl_garden.training.offline.registry"),
}

EPISODE_LENGTH = 8
IMAGE_ENV_ID = "RlgEntrypointBuildImage-v0"
PRIVILEGED_ENV_ID = "RlgEntrypointBuildPrivileged-v0"
IMAGE_ONLY_ENV_ID = "RlgEntrypointBuildImageOnly-v0"
IMAGE_SIZE = 32
IMAGE_ONLY_SIZE = 64
# Uni-O4's OPE picks its termination function from the task name.
LOCOMOTION_NAMED_PENDULUM_ID = "rlg-entrypoint-build-halfcheetah-pendulum-v0"


class _SmallObservationEnv(gym.Env):
    """Observation keys ``state`` (3), ``state_priv`` (2) and ``rgb_<camera>``,
    each only when asked for."""

    def __init__(self, *, state: bool = True, priv: bool = False,
                 camera: str | None = None, image_size: int = IMAGE_SIZE) -> None:
        shapes = {}
        if state:
            shapes["state"] = (3,)
        if priv:
            shapes["state_priv"] = (2,)
        self._shapes = shapes
        self._image_key = f"rgb_{camera}" if camera else None
        self._image_shape = (image_size, image_size, 3)
        spaces_ = {key: spaces.Box(-np.inf, np.inf, shape, np.float32) for key, shape in shapes.items()}
        if self._image_key:
            spaces_[self._image_key] = spaces.Box(0, 255, self._image_shape, np.uint8)
        self.observation_space = spaces.Dict(spaces_)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self._observation(), {}

    def step(self, action):
        self._steps += 1
        return self._observation(), 0.0, False, self._steps >= EPISODE_LENGTH, {}

    def _observation(self):
        observation = {
            key: self.np_random.standard_normal(shape).astype(np.float32)
            for key, shape in self._shapes.items()
        }
        if self._image_key:
            observation[self._image_key] = self.np_random.integers(0, 256, self._image_shape, np.uint8)
        return observation


if LOCOMOTION_NAMED_PENDULUM_ID not in gym.registry:
    gym.register(
        LOCOMOTION_NAMED_PENDULUM_ID,
        entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv",
        max_episode_steps=EPISODE_LENGTH,
    )
for _env_id, _kwargs in (
    (IMAGE_ENV_ID, {"camera": "front"}),
    (PRIVILEGED_ENV_ID, {"priv": True}),
    (IMAGE_ONLY_ENV_ID, {"state": False, "camera": "default", "image_size": IMAGE_ONLY_SIZE}),
):
    if _env_id not in gym.registry:
        gym.register(_env_id, entry_point=_SmallObservationEnv, kwargs=_kwargs)

# Flag name -> value, applied to each algorithm whose Args class has the field
# (the first segment of a dotted flag). A bool is a ``--flag`` / ``--no-flag``.
SMALL_VALUES: dict[str, str | bool] = {
    "env_backend": "gymnasium",
    "env_id": "Pendulum-v1",
    "gymnasium.device": "cpu",
    "gymnasium.import_modules": __name__,
    "num_envs": "2",
    "num_eval_envs": "1",
    "buffer_size": "320",
    "buffer_device": "cpu",
    "batch_size": "8",
    "learning_starts": "8",
    "hidden_dim": "16",
    "num_minibatches": "2",
    "eval_freq": "0",
    "log_type": "none",
    "std_log": False,
    "capture_video": False,
}
EVAL_VALUES = {"eval_freq": "4", "num_eval_steps": "4", "num_eval_episodes": "1"}
IMAGE_VALUES = {"env_id": IMAGE_ENV_ID, "obs.rgb": "front"}

# (phase, algorithm) -> flag overrides on top of SMALL_VALUES.
OVERRIDES: dict[tuple[str, str], dict[str, str | bool]] = {
    ("online", "td3"): IMAGE_VALUES,
    ("online", "drqv2"): IMAGE_VALUES,
    ("online", "tdmpc2"): {"num_envs": "1"},
    ("online", "dreamer_v3"): {"batch_length": "2"},
    # The contract allows one state vector per role: Pendulum's "state" trains the
    # teacher, the student reads the other state key.
    ("online", "policy_distillation"): {
        "env_id": PRIVILEGED_ENV_ID,
        "obs.extra_state": "priv",
        "teacher_obs_keys": "state",
        "student_obs_keys": "state_priv",
        "net_arch": "16 16",
    },
    ("online", "ncrl"): {
        "env_id": IMAGE_ONLY_ENV_ID,
        "obs.rgb": "default",
        "obs.image_size": "None",
        "episode_frames": str(EPISODE_LENGTH),
        "action_repeat": "1",
        "warmup_frames": "0",
        "online_ratio": "1.0",
        "learning_starts": "16",
        "batch_length": "4",
    },
    ("offline", "rssm_pretrain"): {"batch_length": "4", "batch_size": "2"},
    # Dry-run only validates that the phase budget exceeds the warmup phases.
    ("offline", "bppo"): {"num_offline_steps": "3000000"},
    ("offline", "unio4"): {"num_offline_steps": "3000000"},
    ("offline", "unio4_ope"): {
        "num_offline_steps": "3000000", "env_id": LOCOMOTION_NAMED_PENDULUM_ID},
}

# Algorithms whose builder refuses to run without a live eval env.
EVAL_ONLY = {("offline", "bppo"), ("offline", "unio4"), ("offline", "unio4_ope")}

# (phase, algorithm) -> why the build cannot run on CPU in the fast lane.
EXEMPTIONS: dict[tuple[str, str], str] = {
    ("online", "gail"): "demonstrations come only from a D4RL dataset (d4rl_legacy backend, mujoco_py)",
}


def _registered(phase: str):
    registry = importlib.import_module(PHASES[phase][1]).registry
    registry.discover()
    return registry.entries()


REGISTERED = {(phase, name) for phase in PHASES for name in _registered(phase)}


def _supports_eval(phase: str, name: str) -> bool:
    fields = {field.name for field in dataclasses.fields(_registered(phase)[name].args_cls)}
    return {"eval_freq", "env_backend"} <= fields


CASES = sorted(
    (phase, name, with_eval)
    for phase, name in REGISTERED - set(EXEMPTIONS)
    for with_eval in (False, True)
    if (with_eval and _supports_eval(phase, name))
    or (not with_eval and (phase, name) not in EVAL_ONLY)
)


def write_dataset(path: Path, *, image_size: int | None = None) -> Path:
    """Pendulum-shaped H5 in the rl-garden trajectory layout; with ``image_size``
    the observation is a group of ``state`` and an ``rgb_front`` image."""
    rng = np.random.default_rng(0)
    length = 24
    with h5py.File(path, "w") as dataset:
        for episode in range(2):
            theta = rng.uniform(-np.pi, np.pi, length + 1)
            velocity = rng.uniform(-1, 1, length + 1)
            observations = np.column_stack((np.cos(theta), np.sin(theta), velocity))
            truncated = np.zeros(length, dtype=bool)
            truncated[-1] = True
            group = dataset.create_group(f"traj_{episode}")
            observations = observations.astype(np.float32)
            if image_size is None:
                group.create_dataset("obs", data=observations)
            else:
                obs_group = group.create_group("obs")
                obs_group.create_dataset("state", data=observations)
                obs_group.create_dataset(
                    "rgb_front",
                    data=rng.integers(0, 256, (length + 1, image_size, image_size, 3), np.uint8),
                )
            group.create_dataset("actions", data=rng.uniform(-1, 1, (length, 1)).astype(np.float32))
            group.create_dataset("rewards", data=-(theta[:-1] ** 2).astype(np.float32))
            group.create_dataset("terminated", data=np.zeros(length, dtype=bool))
            group.create_dataset("truncated", data=truncated)
    return path


def write_sequence_dataset(root: Path) -> Path:
    """Converted NCRL image-sequence dataset (one embodiment, 64x64 frames)."""
    lengths, action_dim = [20, 20], 2
    root.mkdir(parents=True)
    steps = sum(lengths)
    manifest = {
        "format": "rl_garden_sequence_v1", "obs_key": "rgb_default",
        "image_shape": [IMAGE_ONLY_SIZE, IMAGE_ONLY_SIZE, 3], "padded_action_dim": action_dim,
        "embodiment": "toy", "action_dim": action_dim,
        "num_episodes": len(lengths), "episode_lengths": lengths,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    np.save(root / "obs.npy", np.random.default_rng(0).integers(
        0, 256, (steps, IMAGE_ONLY_SIZE, IMAGE_ONLY_SIZE, 3), np.uint8))
    torch.save(
        {"action": torch.rand(steps, action_dim) * 2 - 1, "reward": torch.rand(steps),
         "episode_id": torch.arange(steps) // 20, "is_first": torch.arange(steps) % 20 == 0},
        root / "episodes.pt",
    )
    return root


def write_multitask_dataset(root: Path) -> Path:
    """TD-MPC2 multitask dataset in the converter's layout (one task)."""
    episodes, length = 3, 12
    (root / "pendulum-swingup").mkdir(parents=True)
    manifest = {
        "tasks": ["pendulum-swingup"], "obs_dims": [3], "action_dims": [1],
        "episode_lengths": [length], "num_episodes": [episodes],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    torch.save(
        {"obs": torch.randn(episodes, length, 3), "action": torch.rand(episodes, length, 1),
         "reward": torch.randn(episodes, length)},
        root / "pendulum-swingup" / "episodes.pt",
    )
    return root


def _flags(args_cls, values: dict[str, str | bool]) -> list[str]:
    fields = {field.name for field in dataclasses.fields(args_cls)}
    argv = []
    for name, value in values.items():
        if name.split(".")[0] not in fields:
            continue
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{flag[2:]}")
        else:
            argv += [flag, *value.split()]
    return argv


def build_argv(
    phase: str, name: str, log_dir: Path, *, with_eval: bool = False,
    inputs: dict[str, str | bool] | None = None,
) -> list[str]:
    args_cls = _registered(phase)[name].args_cls
    values = {**SMALL_VALUES, "log_dir": str(log_dir)}
    if with_eval:
        values.update(EVAL_VALUES)
    values.update(OVERRIDES.get((phase, name), {}))
    values.update(inputs or {})
    return [name, *_flags(args_cls, values)]


class SharedInputs:
    """Datasets and checkpoints some algorithms consume, each made once per
    module by really running the producing entrypoint for a couple of steps."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _train(self, phase: str, name: str, inputs: dict[str, str | bool]) -> Path:
        directory = self.root / f"{phase}_{name}"
        argv = build_argv(
            phase, name, directory / "runs",
            inputs={"checkpoint_dir": str(directory / "checkpoints"), **inputs},
        )
        result = run_entry(PHASES[phase][0], argv)
        assert result.returncode == 0, f"producing {phase}/{name}: {result.stderr[-1500:]!r}"
        (checkpoint,) = (directory / "checkpoints").glob("*.pt")
        return checkpoint

    @functools.cached_property
    def h5(self) -> Path:
        return write_dataset(self.root / "dataset.h5")

    @functools.cached_property
    def image_h5(self) -> Path:
        return write_dataset(self.root / "image_dataset.h5", image_size=IMAGE_SIZE)

    @functools.cached_property
    def sequence_dir(self) -> Path:
        return write_sequence_dataset(self.root / "sequence")

    @functools.cached_property
    def multitask_dir(self) -> Path:
        return write_multitask_dataset(self.root / "multitask")

    @functools.cached_property
    def opal_checkpoint(self) -> Path:
        return self._train("offline", "opal", {"dataset_path": str(self.h5), "num_offline_steps": "2"})

    @functools.cached_property
    def iql_guide_checkpoint(self) -> Path:
        return self._train(
            "offline", "iql",
            {"offline_dataset": str(self.h5), "action_low": "-2", "action_high": "2",
             "num_offline_steps": "2", "actor_use_layer_norm": False},
        )

    @functools.cached_property
    def diffusion_bc_checkpoint(self) -> Path:
        return self._train("offline", "diffusion_bc",
                           {"dataset_path": str(self.h5), "num_offline_steps": "2"})

    @functools.cached_property
    def rssm_checkpoint(self) -> Path:
        return self._train(
            "offline", "rssm_pretrain",
            {"dataset_dir": str(self.sequence_dir), "num_offline_steps": "2"},
        )

    @functools.cached_property
    def ppo_teacher_checkpoint(self) -> Path:
        return self._train(
            "online", "ppo", {"num_steps": "4", "total_timesteps": "16", "net_arch": "16 16"},
        )

    @functools.cached_property
    def world_model_checkpoint(self) -> Path:
        return self._train(
            "offline", "tdmpc2_multitask",
            {"dataset_dir": str(self.multitask_dir), "mmap_dir": str(self.root / "mt_mmap"),
             "num_offline_steps": "2"},
        )


@pytest.fixture(scope="module")
def shared_inputs(tmp_path_factory):
    return SharedInputs(tmp_path_factory.mktemp("entrypoint_build_inputs"))


# (phase, algorithm) -> the dataset/checkpoint flags it cannot build without.
INPUTS: dict[tuple[str, str], Callable[[SharedInputs, Path], dict[str, str | bool]]] = {
    ("online", "supe"): lambda s, t: {"opal_checkpoint": str(s.opal_checkpoint)},
    ("online", "jsrl"): lambda s, t: {"guide_checkpoint": str(s.iql_guide_checkpoint),
                                      "max_horizon": "10"},
    ("online", "policy_distillation"): lambda s, t: {
        "teacher_checkpoint": str(s.ppo_teacher_checkpoint)},
    ("online", "ncrl"): lambda s, t: {"pretrained_world_model": str(s.rssm_checkpoint)},
    ("offline", "a2a_bc"): lambda s, t: {"dataset_path": str(s.image_h5), "obs.rgb": "front"},
    ("offline", "consistency_distill_bc"): lambda s, t: {
        "dataset_path": str(s.h5), "bc_checkpoint": str(s.diffusion_bc_checkpoint)},
    ("offline", "rssm_pretrain"): lambda s, t: {"dataset_dir": str(s.sequence_dir)},
    ("offline", "pwm"): lambda s, t: {
        "dataset_dir": str(s.multitask_dir), "mmap_dir": str(t / "mmap"),
        "task": "pendulum-swingup", "world_model_checkpoint": str(s.world_model_checkpoint)},
    ("offline", "tdmpc2_multitask"): lambda s, t: {
        "dataset_dir": str(s.multitask_dir), "mmap_dir": str(t / "mmap")},
}
for _name in ("diffusion_bc", "hilp", "opal"):
    INPUTS[("offline", _name)] = lambda s, t: {"dataset_path": str(s.h5)}


def inputs_for(phase, name, shared, tmp_path):
    if (phase, name) in INPUTS:
        return INPUTS[(phase, name)](shared, tmp_path)
    if phase == "online":
        return {}
    return {"offline_dataset": str(shared.h5), "action_low": "-2", "action_high": "2"}


@pytest.fixture(autouse=True)
def cpu_only_and_restored_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    yield
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def test_registry_and_tables_agree():
    assert EXEMPTIONS.keys() <= REGISTERED, "exemption for an unregistered algorithm"
    assert (INPUTS.keys() | OVERRIDES.keys() | EVAL_ONLY) <= REGISTERED
    assert {(phase, name) for phase, name, _ in CASES} == REGISTERED - EXEMPTIONS.keys()
    assert len(REGISTERED) >= 70


@pytest.mark.parametrize(
    "phase,name,with_eval", CASES,
    ids=[f"{p}-{n}-{'eval_env' if e else 'no_eval'}" for p, n, e in CASES],
)
def test_dry_run_builds_env_and_agent(phase, name, with_eval, tmp_path, shared_inputs):
    inputs = inputs_for(phase, name, shared_inputs, tmp_path)
    argv = build_argv(phase, name, tmp_path / "runs", with_eval=with_eval, inputs=inputs)
    result = run_entry(PHASES[phase][0], [*argv, "--dry-run"])
    assert result.returncode == 0, f"{result.stderr[-1500:]!r} {result.stdout[-500:]!r}"


@pytest.mark.parametrize("phase,name", sorted(EXEMPTIONS), ids=lambda key: str(key))
def test_exempted_algorithm_still_cannot_build(phase, name, tmp_path, shared_inputs):
    inputs = inputs_for(phase, name, shared_inputs, tmp_path)
    argv = build_argv(phase, name, tmp_path / "runs", inputs=inputs)
    try:
        result = run_entry(PHASES[phase][0], [*argv, "--dry-run"])
    except Exception:
        return
    assert result.returncode != 0, f"{phase}/{name} builds now; drop its exemption"
