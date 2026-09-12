"""Tests for the ``maniskill`` env backend: per-camera rgb/depth keys, the
resolved camera config, and the observation contract."""
from __future__ import annotations

import pytest
import torch

from rl_garden.envs.backend_registry import EnvRequest, discover_env_backends, _REGISTRY
from rl_garden.envs.backends.maniskill import ManiSkillBackend
from rl_garden.envs.maniskill import make_maniskill_env
from rl_garden.envs.maniskill.config import ManiSkillEnvConfig
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError

# PickCube-v1's single sensor, per mani_skill.envs.tasks.tabletop.pick_cube
# (CameraConfig("base_camera", ...)).
_CAMERA = "base_camera"


def test_backend_registers_with_api_version_2():
    discover_env_backends()
    assert "maniskill" in _REGISTRY
    assert _REGISTRY["maniskill"] is ManiSkillBackend
    assert ManiSkillBackend.api_version == 2
    assert ManiSkillBackend.config_field == "maniskill"


def _make_req(**overrides):
    defaults = dict(
        env_id="PickCube-v1",
        num_envs=2,
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        seed=0,
        observation=ObservationConfig(),
        num_eval_envs=2,
        backend_config=None,
    )
    defaults.update(overrides)
    return EnvRequest(**defaults)


def test_resolve_config_maps_observation_to_rgb_depth_cameras_and_image_size():
    req = _make_req(
        observation=ObservationConfig(rgb=(_CAMERA,), depth=(_CAMERA,), image_size=(64, 64))
    )
    cfg = ManiSkillBackend.resolve_config(req, is_eval=False)
    assert isinstance(cfg, ManiSkillEnvConfig)
    assert cfg.rgb_cameras == (_CAMERA,)
    assert cfg.depth_cameras == (_CAMERA,)
    assert cfg.camera_height == 64
    assert cfg.camera_width == 64
    assert cfg.state is True


def test_resolve_config_state_only_has_no_cameras():
    req = _make_req()
    cfg = ManiSkillBackend.resolve_config(req, is_eval=False)
    assert cfg.rgb_cameras == ()
    assert cfg.depth_cameras == ()
    assert cfg.camera_height is None
    assert cfg.camera_width is None


def test_resolve_config_splits_train_eval_num_envs():
    req = _make_req(num_envs=8, num_eval_envs=4)
    assert ManiSkillBackend.resolve_config(req, is_eval=False).num_envs == 8
    assert ManiSkillBackend.resolve_config(req, is_eval=True).num_envs == 4


def test_state_only_env_is_dict_with_state_key():
    cfg = ManiSkillEnvConfig(env_id="PickCube-v1", num_envs=2)
    env = make_maniskill_env(cfg)
    try:
        assert set(env.single_observation_space.spaces) == {"state"}
        obs, _ = env.reset(seed=0)
        assert isinstance(obs["state"], torch.Tensor)
    finally:
        env.close()


def test_vision_env_produces_named_per_camera_keys_no_channel_stacking():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1",
        num_envs=2,
        rgb_cameras=(_CAMERA,),
        depth_cameras=(_CAMERA,),
        state=True,
    )
    env = make_maniskill_env(cfg)
    try:
        keys = set(env.single_observation_space.spaces)
        assert keys == {"state", f"rgb_{_CAMERA}", f"depth_{_CAMERA}"}
        assert env.single_observation_space[f"rgb_{_CAMERA}"].dtype.name == "uint8"
        obs, _ = env.reset(seed=0)
        assert set(obs) == keys
    finally:
        env.close()


def test_vision_env_state_false_drops_state_key():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1",
        num_envs=2,
        rgb_cameras=(_CAMERA,),
        state=False,
    )
    env = make_maniskill_env(cfg)
    try:
        assert "state" not in env.single_observation_space.spaces
        assert set(env.single_observation_space.spaces) == {f"rgb_{_CAMERA}"}
    finally:
        env.close()


def test_unknown_camera_raises_and_lists_available():
    cfg = ManiSkillEnvConfig(
        env_id="PickCube-v1",
        num_envs=1,
        rgb_cameras=("not_a_real_camera",),
    )
    with pytest.raises(ObservationContractError, match="not_a_real_camera"):
        make_maniskill_env(cfg)
