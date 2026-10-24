"""Tests for the ``dm_control`` env backend.

The registry/config/hard-reward tests need no simulator. Tests that build a
real env skip when ``dm_control`` is not installed; the pixel test also
skips without a GL backend (set ``MUJOCO_GL=egl`` on headless servers).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_garden.common.env_args import DMControlConfig
from rl_garden.envs.backend_registry import EnvRequest
from rl_garden.envs.backends.dm_control import DMControlBackend
from rl_garden.envs.dm_control.env import (
    DEFAULT_CAMERA_NAME,
    HARD_ACTION_PENALTY,
    hard_reward_threshold,
    hard_task_reward,
)
from rl_garden.envs.dm_control.tasks import (
    NCRL_CUSTOM_DMC_TASKS,
    NCRL_DMC_TASKS,
    parse_task_id,
)
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


def _make_req(**overrides):
    defaults = {
        "env_id": "walker-walk",
        "num_envs": 2,
        "control_mode": "n/a",
        "render_mode": "rgb_array",
        "seed": 1,
        "observation": ObservationConfig(),
        "num_eval_envs": 1,
        "create_eval_env": False,
        "backend_config": DMControlConfig(),
    }
    defaults.update(overrides)
    return EnvRequest(**defaults)


# --- no simulator needed ----------------------------------------------------


def test_backend_registers_with_api_version_2():
    from rl_garden.envs.backend_registry import _REGISTRY, discover_env_backends

    discover_env_backends()
    assert _REGISTRY["dm_control"] is DMControlBackend
    assert DMControlBackend.api_version == 2
    assert DMControlBackend.config_field == "dm_control"


def test_discovery_and_config_do_not_import_dm_control():
    repo_root = Path(__file__).resolve().parents[1]
    command = """
import sys
from rl_garden.envs.backend_registry import discover_env_backends
from rl_garden.envs.backends.dm_control import DMControlBackend
from rl_garden.envs.dm_control import DMControlEnvConfig, make_dm_control_env
discover_env_backends()
assert 'dm_control' not in sys.modules, 'dm_control imported eagerly'
"""
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        check=True,
        env=os.environ | {"MPLCONFIGDIR": "/tmp"},
    )


@pytest.mark.parametrize(
    ("task_id", "expected"),
    [
        ("walker-walk", ("walker", "walk", False)),
        ("walker-run-backwards", ("walker", "run_backwards", False)),
        ("walker-walk-hard", ("walker", "walk", True)),
        ("acrobot-swingup-hard", ("acrobot", "swingup", True)),
        ("dmc-walker-walk-hard", ("walker", "walk", True)),
        ("reacher-hard", ("reacher", "hard", False)),
        ("finger-turn-hard", ("finger", "turn_hard", False)),
        ("dmc-acrobot-swingup-sparse", ("acrobot", "swingup_sparse", False)),
        ("quadruped-roll-fast", ("quadruped", "roll_fast", False)),
        ("cup-catch", ("ball_in_cup", "catch", False)),
    ],
)
def test_parse_task_id(task_id, expected):
    assert parse_task_id(task_id) == expected


def test_parse_task_id_rejects_missing_task():
    with pytest.raises(ValueError):
        parse_task_id("walker")


def test_ncrl_task_set_is_copied_verbatim():
    assert len(NCRL_DMC_TASKS) == 22
    assert len(NCRL_CUSTOM_DMC_TASKS) == 11
    assert set(NCRL_CUSTOM_DMC_TASKS) <= set(NCRL_DMC_TASKS)
    assert {task.split("-")[0] for task in NCRL_CUSTOM_DMC_TASKS} == {"cheetah", "walker", "quadruped"}


def test_resolve_config_defaults_and_train_eval_split():
    req = _make_req(num_envs=3, num_eval_envs=2)

    train_cfg = DMControlBackend.resolve_config(req, is_eval=False)
    eval_cfg = DMControlBackend.resolve_config(req, is_eval=True)

    assert train_cfg.num_envs == 3 and eval_cfg.num_envs == 2
    assert train_cfg.action_repeat == 2
    assert train_cfg.image_size == (64, 64)
    assert train_cfg.camera_id is None
    assert train_cfg.hard is False
    assert train_cfg.state is True and train_cfg.rgb_cameras == ()


def test_resolve_config_suite_hard_tasks_need_explicit_flag():
    for env_id in ("reacher-hard", "finger-turn-hard"):
        assert not DMControlBackend.resolve_config(_make_req(env_id=env_id), is_eval=False).hard
    req = _make_req(env_id="reacher-hard", backend_config=DMControlConfig(hard=True))
    assert DMControlBackend.resolve_config(req, is_eval=False).hard


def test_suite_hard_tasks_construct():
    _require_dm_control()
    from rl_garden.envs.dm_control.env import DMControlGymEnv

    for env_id in ("reacher-hard", "finger-turn-hard"):
        env = DMControlGymEnv(env_id, seed=0)
        assert env._hard is False
        env.reset(seed=0)
    assert DMControlGymEnv("reacher-hard", seed=0, hard=True)._hard is True


def test_resolve_config_hard_suffix_and_flag():
    assert DMControlBackend.resolve_config(_make_req(env_id="walker-walk-hard"), is_eval=False).hard
    req = _make_req(backend_config=DMControlConfig(hard=True, action_repeat=4, camera_id=1))
    cfg = DMControlBackend.resolve_config(req, is_eval=False)
    assert cfg.hard and cfg.action_repeat == 4 and cfg.camera_id == 1


def test_resolve_config_pixels():
    obs = ObservationConfig(state=False, rgb=(DEFAULT_CAMERA_NAME,), image_size=(32, 48))
    cfg = DMControlBackend.resolve_config(_make_req(observation=obs), is_eval=False)
    assert cfg.rgb_cameras == ("default",) and cfg.state is False and cfg.image_size == (32, 48)


@pytest.mark.parametrize(
    "observation",
    [
        ObservationConfig(rgb=("side",)),
        ObservationConfig(depth=(DEFAULT_CAMERA_NAME,)),
        ObservationConfig(extra_state=("pose",)),
        ObservationConfig(frame_stack=3),
    ],
)
def test_resolve_config_rejects_unsupported_observations(observation):
    with pytest.raises(ObservationContractError):
        DMControlBackend.resolve_config(_make_req(observation=observation), is_eval=False)


def test_hard_task_reward_threshold_and_penalty():
    action = np.array([0.6, 0.8])  # ||a|| = 1
    assert hard_task_reward(0.5, action, reward_threshold=0.6) == pytest.approx(-0.1)
    assert hard_task_reward(0.7, action, reward_threshold=0.6) == pytest.approx(0.6)
    assert hard_task_reward(0.7, np.zeros(2), reward_threshold=0.6) == pytest.approx(0.7)
    assert HARD_ACTION_PENALTY == 0.1
    assert hard_reward_threshold("cheetah") == 0.2
    assert hard_reward_threshold("walker") == 0.6
    assert hard_reward_threshold("quadruped") == 0.6


# --- real dm_control --------------------------------------------------------


def _require_dm_control():
    pytest.importorskip("dm_control")


def test_state_observation_contract_through_registry():
    _require_dm_control()
    from rl_garden.envs.backend_registry import make_training_envs

    train_env, eval_env = make_training_envs("dm_control", _make_req(create_eval_env=True))

    assert train_env.num_envs == 2 and eval_env.num_envs == 1
    space = train_env.single_observation_space
    assert set(space.spaces) == {"state"}
    assert space["state"].dtype == np.float32 and space["state"].shape == (24,)
    action_space = train_env.single_action_space
    assert action_space.shape == (6,)
    assert np.all(action_space.low == -1.0) and np.all(action_space.high == 1.0)

    obs, _ = train_env.reset(seed=0)
    assert isinstance(obs["state"], torch.Tensor)
    assert obs["state"].shape == (2, 24) and obs["state"].dtype == torch.float32
    next_obs, reward, terminated, truncated, _ = train_env.step(
        torch.as_tensor(train_env.action_space.sample())
    )
    assert next_obs["state"].shape == (2, 24)
    assert reward.shape == (2,) and terminated.dtype == torch.bool and truncated.dtype == torch.bool
    train_env.close()
    eval_env.close()


def test_pixel_observation_contract():
    _require_dm_control()
    if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
        pytest.skip("no GL backend: set MUJOCO_GL=egl")
    from rl_garden.envs.backend_registry import make_training_envs

    obs_cfg = ObservationConfig(state=True, rgb=(DEFAULT_CAMERA_NAME,))
    train_env, _ = make_training_envs("dm_control", _make_req(observation=obs_cfg))

    space = train_env.single_observation_space
    assert set(space.spaces) == {"rgb_default", "state"}
    assert space["rgb_default"].dtype == np.uint8
    assert space["rgb_default"].shape == (64, 64, 3)
    obs, _ = train_env.reset(seed=0)
    assert obs["rgb_default"].dtype == torch.uint8
    assert obs["rgb_default"].shape == (2, 64, 64, 3)
    assert obs["rgb_default"].float().std() > 0  # a real render, not a blank frame
    train_env.close()


def test_pixel_frame_stack_enables_autoreset_in_step():
    _require_dm_control()
    if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
        pytest.skip("no GL backend: set MUJOCO_GL=egl")
    from rl_garden.envs.backend_registry import make_training_envs

    obs_cfg = ObservationConfig(state=True, rgb=(DEFAULT_CAMERA_NAME,), frame_stack=3)
    train_env, _ = make_training_envs("dm_control", _make_req(observation=obs_cfg))

    assert train_env.autoreset_in_step
    assert train_env.single_observation_space["rgb_default"].shape == (3, 64, 64, 3)
    train_env.close()


def test_action_repeat_gives_500_step_episodes():
    _require_dm_control()
    from rl_garden.envs.dm_control.env import DMControlGymEnv

    env = DMControlGymEnv("walker-walk", seed=0, action_repeat=2)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == 500


def test_reset_seed_is_deterministic():
    _require_dm_control()
    from rl_garden.envs.dm_control.env import DMControlGymEnv

    env = DMControlGymEnv("quadruped-walk", seed=0)
    first, _ = env.reset(seed=7)
    env.step(env.action_space.sample())
    second, _ = env.reset(seed=7)
    np.testing.assert_array_equal(first["state"], second["state"])


def test_hard_variant_applies_per_frame_inside_action_repeat():
    _require_dm_control()
    import dm_env

    from rl_garden.envs.dm_control.env import DMControlGymEnv

    env = DMControlGymEnv("walker-walk-hard", seed=0, action_repeat=2)
    env.reset(seed=0)
    real_step = env._env.step
    fake_rewards = iter([0.5, 0.7])

    def fake_step(action):
        time_step = real_step(action)
        return dm_env.TimeStep(time_step.step_type, next(fake_rewards), time_step.discount, time_step.observation)

    env._env.step = fake_step
    action = np.array([0.6, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # ||a|| = 1
    _, reward, _, _, _ = env.step(action)
    # frame 1: 0.5 < 0.6 -> 0 - 0.1; frame 2: 0.7 - 0.1.
    assert reward == pytest.approx(-0.1 + 0.6)


@pytest.mark.parametrize("task_id", NCRL_DMC_TASKS)
def test_every_ncrl_task_constructs_and_steps(task_id):
    _require_dm_control()
    from rl_garden.envs.dm_control.env import DMControlGymEnv

    env = DMControlGymEnv(task_id, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs["state"].dtype == np.float32 and obs["state"].ndim == 1
    next_obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert np.isfinite(reward) and not terminated and not truncated
    assert next_obs["state"].shape == obs["state"].shape
