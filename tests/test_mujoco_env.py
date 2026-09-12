"""Tests for the ``mujoco`` env backend (Gymnasium benchmark tasks +
rl-garden's own custom tasks, SyncVectorEnv/AsyncVectorEnv + torch adapter)."""
from __future__ import annotations

import pytest
import torch

from rl_garden.envs.backend_registry import EnvRequest, discover_env_backends, _REGISTRY
from rl_garden.envs.backends.mujoco import MujocoBackend
from rl_garden.envs.mujoco import make_mujoco_env
from rl_garden.envs.mujoco.config import MujocoEnvConfig
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError


def test_backend_registers_with_api_version_2():
    discover_env_backends()
    assert "mujoco" in _REGISTRY
    assert _REGISTRY["mujoco"] is MujocoBackend
    assert MujocoBackend.api_version == 2
    assert MujocoBackend.config_field == "mujoco"


def _make_req(**overrides):
    defaults = dict(
        env_id="InvertedPendulum-v4",
        num_envs=2,
        control_mode="n/a",
        render_mode="rgb_array",
        seed=0,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    defaults.update(overrides)
    return EnvRequest(**defaults)


def test_resolve_config_is_side_effect_free_and_splits_train_eval_num_envs():
    req = _make_req(num_envs=4, num_eval_envs=2)

    train_cfg = MujocoBackend.resolve_config(req, is_eval=False)
    assert isinstance(train_cfg, MujocoEnvConfig)
    assert train_cfg.num_envs == 4

    eval_cfg = MujocoBackend.resolve_config(req, is_eval=True)
    assert eval_cfg.num_envs == 2


def test_resolve_config_rejects_vision_request():
    req = _make_req(observation=ObservationConfig(rgb=("cam",)))

    with pytest.raises(ObservationContractError):
        MujocoBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_image_size():
    req = _make_req(observation=ObservationConfig(image_size=(64, 64)))

    with pytest.raises(ObservationContractError):
        MujocoBackend.resolve_config(req, is_eval=False)


def test_resolve_config_rejects_frame_stack():
    req = _make_req(observation=ObservationConfig(frame_stack=2))

    with pytest.raises(ObservationContractError):
        MujocoBackend.resolve_config(req, is_eval=False)


def test_gymnasium_benchmark_task_observation_is_dict_with_state_key():
    cfg = MujocoEnvConfig(env_id="InvertedPendulum-v4", num_envs=2, seed=0, device="cpu")
    env = make_mujoco_env(cfg)
    try:
        assert set(env.single_observation_space.spaces) == {"state"}
        obs, _ = env.reset(seed=0)
        assert isinstance(obs["state"], torch.Tensor)
        assert obs["state"].shape[0] == 2

        actions = torch.as_tensor(env.action_space.sample())
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert set(next_obs) == {"state"}
        assert isinstance(rewards, torch.Tensor) and rewards.dtype == torch.float32
    finally:
        env.close()


def test_custom_task_observation_is_already_dict_and_not_double_wrapped():
    import rl_garden.envs.mujoco.tasks  # noqa: F401 -- registers the custom task ids.

    cfg = MujocoEnvConfig(
        env_id="RlGarden-InvertedPendulum-Custom-v0", num_envs=2, seed=0, device="cpu"
    )
    env = make_mujoco_env(cfg)
    try:
        assert set(env.single_observation_space.spaces) == {"state"}
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"state"}
        assert obs["state"].shape == (2, 4)
    finally:
        env.close()


def test_make_train_and_eval_env_construct_working_vectorized_envs():
    req = _make_req(num_envs=2, num_eval_envs=2)

    train_env = MujocoBackend.make_train_env(req)
    eval_env = MujocoBackend.make_eval_env(req)
    try:
        assert train_env.num_envs == 2
        assert eval_env.num_envs == 2
        train_env.reset(seed=0)
        eval_env.reset(seed=0)
    finally:
        train_env.close()
        eval_env.close()
