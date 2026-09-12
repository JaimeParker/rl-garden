"""Tests for rl_garden.observations: config, schema, groups."""
from __future__ import annotations

import numpy as np
import pytest
import tyro
from gymnasium import spaces

from rl_garden.observations import (
    Modality,
    ObsGroups,
    ObservationConfig,
    ObservationContractError,
    ObservationSchema,
    key_modality,
    normalize_observation_space,
    resolve_obs_groups,
    validate_observation_space,
)


# ---------------------------------------------------------------------------
# ObservationConfig
# ---------------------------------------------------------------------------


def test_config_defaults_state_only():
    cfg = ObservationConfig()
    assert cfg.state is True
    assert not cfg.is_visual
    assert cfg.image_keys == ()
    assert cfg.expected_keys == ("state",)


def test_config_image_keys_rgb_then_depth():
    cfg = ObservationConfig(rgb=("base_camera",), depth=("wrist",), state=True)
    assert cfg.image_keys == ("rgb_base_camera", "depth_wrist")
    assert cfg.expected_keys == ("rgb_base_camera", "depth_wrist", "state")
    assert cfg.is_visual


def test_config_no_state_visual_only():
    cfg = ObservationConfig(state=False, rgb=("base_camera",))
    assert cfg.expected_keys == ("rgb_base_camera",)


def test_config_coerces_lists_to_tuples():
    cfg = ObservationConfig(rgb=["base_camera", "wrist"], image_size=[64, 64])
    assert cfg.rgb == ("base_camera", "wrist")
    assert isinstance(cfg.rgb, tuple)
    assert cfg.image_size == (64, 64)
    assert isinstance(cfg.image_size, tuple)


def test_config_rejects_frame_stack_below_one():
    with pytest.raises(ValueError):
        ObservationConfig(frame_stack=0)


def test_config_rejects_no_modality():
    with pytest.raises(ValueError):
        ObservationConfig(state=False, rgb=(), depth=())


def test_config_rejects_non_positive_image_size():
    with pytest.raises(ValueError):
        ObservationConfig(rgb=("cam",), image_size=(0, 64))


def test_config_rejects_bad_camera_name():
    with pytest.raises(ValueError):
        ObservationConfig(rgb=("bad/name",))
    with pytest.raises(ValueError):
        ObservationConfig(rgb=("",))


def test_config_tyro_round_trip():
    cfg = tyro.cli(
        ObservationConfig, args=["--rgb", "base_camera", "--image-size", "64", "64"]
    )
    assert cfg.rgb == ("base_camera",)
    assert cfg.image_size == (64, 64)
    assert cfg.state is True


# ---------------------------------------------------------------------------
# key_modality
# ---------------------------------------------------------------------------


def test_key_modality():
    assert key_modality("state") == Modality.STATE
    assert key_modality("rgb_base_camera") == Modality.RGB
    assert key_modality("depth_wrist") == Modality.DEPTH


def test_key_modality_rejects_unknown():
    with pytest.raises(ObservationContractError):
        key_modality("proprio")
    with pytest.raises(ObservationContractError):
        key_modality("camera")


def test_key_modality_rejects_bare_rgb_depth():
    with pytest.raises(ObservationContractError):
        key_modality("rgb")
    with pytest.raises(ObservationContractError):
        key_modality("depth")


# ---------------------------------------------------------------------------
# ObservationSchema.from_space
# ---------------------------------------------------------------------------


def _state_box(dim=10):
    return spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)


def _rgb_box(h=64, w=64, stacked=False):
    shape = (2, h, w, 3) if stacked else (h, w, 3)
    return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)


def _depth_box(h=64, w=64, stacked=False):
    shape = (2, h, w, 1) if stacked else (h, w, 1)
    return spaces.Box(low=0.0, high=np.inf, shape=shape, dtype=np.float32)


def test_from_space_box_gives_single_state_entry():
    schema = ObservationSchema.from_space(_state_box(10))
    assert schema.keys == ("state",)
    assert schema.state_key == "state"
    assert not schema.has_images
    assert schema.has_state
    assert schema.entries["state"].modality == Modality.STATE
    assert schema.entries["state"].shape == (10,)
    assert schema.entries["state"].stacked is False


def test_from_space_dict_one_entry_per_key():
    space = spaces.Dict(
        {"state": _state_box(), "rgb_base_camera": _rgb_box(), "depth_wrist": _depth_box()}
    )
    schema = ObservationSchema.from_space(space)
    assert set(schema.keys) == {"state", "rgb_base_camera", "depth_wrist"}
    assert schema.image_keys == ("rgb_base_camera", "depth_wrist")
    assert schema.has_images
    assert schema.has_state


def test_from_space_stacked_via_4d_shape():
    space = spaces.Dict({"rgb_base_camera": _rgb_box(stacked=True)})
    schema = ObservationSchema.from_space(space)
    assert schema.entries["rgb_base_camera"].stacked is True


def test_from_space_stacked_via_frame_stack_arg():
    space = spaces.Dict({"rgb_base_camera": _rgb_box(stacked=False)})
    schema = ObservationSchema.from_space(space, frame_stack=2)
    assert schema.entries["rgb_base_camera"].stacked is True


def test_from_space_state_never_marked_stacked():
    space = spaces.Dict({"state": _state_box()})
    schema = ObservationSchema.from_space(space, frame_stack=4)
    assert schema.entries["state"].stacked is False


def test_from_space_rejects_unknown_key():
    space = spaces.Dict({"proprio": _state_box()})
    with pytest.raises(ObservationContractError):
        ObservationSchema.from_space(space)


def test_from_space_rejects_bad_type():
    with pytest.raises(ObservationContractError):
        ObservationSchema.from_space(spaces.Discrete(4))


def test_schema_image_keys_order_rgb_then_depth():
    space = spaces.Dict(
        {"depth_wrist": _depth_box(), "rgb_base_camera": _rgb_box(), "state": _state_box()}
    )
    schema = ObservationSchema.from_space(space)
    assert schema.image_keys == ("rgb_base_camera", "depth_wrist")


def test_schema_subset():
    space = spaces.Dict({"state": _state_box(), "rgb_base_camera": _rgb_box()})
    schema = ObservationSchema.from_space(space)
    sub = schema.subset(("state",))
    assert sub.keys == ("state",)
    assert not sub.has_images


def test_schema_subset_unknown_key_raises():
    space = spaces.Dict({"state": _state_box()})
    schema = ObservationSchema.from_space(space)
    with pytest.raises(ObservationContractError):
        schema.subset(("rgb_base_camera",))


def test_schema_image_size():
    space = spaces.Dict({"rgb_base_camera": _rgb_box(h=48, w=64)})
    schema = ObservationSchema.from_space(space)
    assert schema.image_size("rgb_base_camera") == (48, 64)


def test_schema_image_size_stacked():
    space = spaces.Dict({"rgb_base_camera": _rgb_box(h=48, w=64, stacked=True)})
    schema = ObservationSchema.from_space(space)
    assert schema.image_size("rgb_base_camera") == (48, 64)


def test_schema_image_size_rejects_state_key():
    space = spaces.Dict({"state": _state_box()})
    schema = ObservationSchema.from_space(space)
    with pytest.raises(ObservationContractError):
        schema.image_size("state")


# ---------------------------------------------------------------------------
# normalize_observation_space / validate_observation_space
# ---------------------------------------------------------------------------


def test_normalize_box_wraps_in_dict():
    box = _state_box()
    space = normalize_observation_space(box)
    assert isinstance(space, spaces.Dict)
    assert set(space.spaces.keys()) == {"state"}


def test_normalize_dict_returned_as_is():
    dict_space = spaces.Dict({"state": _state_box(), "rgb_base_camera": _rgb_box()})
    space = normalize_observation_space(dict_space)
    assert space is dict_space


def test_validate_rejects_non_dict_non_box():
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Discrete(4))


def test_validate_rejects_unknown_key():
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"proprio": _state_box()}))


def test_validate_rejects_state_not_1d():
    bad = spaces.Box(low=-1.0, high=1.0, shape=(2, 3), dtype=np.float32)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"state": bad}))
    with pytest.raises(ObservationContractError):
        validate_observation_space(bad)


def test_validate_accepts_state_float64():
    # Plain gymnasium envs are commonly float64; backends/datasets cast to
    # float32 themselves before data reaches a buffer or encoder.
    ok = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)
    validate_observation_space(spaces.Dict({"state": ok}))
    validate_observation_space(ok)


def test_validate_rejects_state_non_float_dtype():
    bad = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.int32)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"state": bad}))


def test_validate_rejects_rgb_not_uint8():
    bad = spaces.Box(low=0.0, high=1.0, shape=(64, 64, 3), dtype=np.float32)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"rgb_base_camera": bad}))


def test_validate_rejects_rgb_wrong_last_dim():
    bad = spaces.Box(low=0, high=255, shape=(64, 64, 4), dtype=np.uint8)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"rgb_base_camera": bad}))


def test_validate_rejects_rgb_wrong_ndim():
    bad = spaces.Box(low=0, high=255, shape=(64 * 64 * 3,), dtype=np.uint8)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"rgb_base_camera": bad}))


def test_validate_accepts_rgb_3d_and_4d():
    validate_observation_space(spaces.Dict({"rgb_base_camera": _rgb_box()}))
    validate_observation_space(spaces.Dict({"rgb_base_camera": _rgb_box(stacked=True)}))


def test_validate_rejects_depth_not_float32():
    bad = spaces.Box(low=0, high=32767, shape=(64, 64, 1), dtype=np.int16)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"depth_wrist": bad}))


def test_validate_rejects_depth_wrong_last_dim():
    bad = spaces.Box(low=0.0, high=np.inf, shape=(64, 64, 3), dtype=np.float32)
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"depth_wrist": bad}))


def test_validate_accepts_depth_3d_and_4d():
    validate_observation_space(spaces.Dict({"depth_wrist": _depth_box()}))
    validate_observation_space(spaces.Dict({"depth_wrist": _depth_box(stacked=True)}))


def test_validate_rejects_nested_dict():
    nested = spaces.Dict({"inner": _state_box()})
    with pytest.raises(ObservationContractError):
        validate_observation_space(spaces.Dict({"state": nested}))


def test_validate_accepts_state_only_box():
    validate_observation_space(_state_box())


# ---------------------------------------------------------------------------
# ObsGroups / resolve_obs_groups
# ---------------------------------------------------------------------------


def _full_schema():
    space = spaces.Dict(
        {"state": _state_box(), "rgb_base_camera": _rgb_box(), "depth_wrist": _depth_box()}
    )
    return ObservationSchema.from_space(space)


def test_obs_groups_symmetric_when_none():
    groups = ObsGroups()
    assert groups.is_symmetric


def test_obs_groups_symmetric_when_equal():
    groups = ObsGroups(actor=("state",), critic=("state",))
    assert groups.is_symmetric


def test_obs_groups_asymmetric():
    groups = ObsGroups(actor=("rgb_base_camera",), critic=("state", "rgb_base_camera"))
    assert not groups.is_symmetric


def test_obs_groups_coerces_lists():
    groups = ObsGroups(actor=["state"], critic=["state"])
    assert groups.actor == ("state",)
    assert isinstance(groups.actor, tuple)


def test_resolve_obs_groups_none_is_full_schema():
    schema = _full_schema()
    resolved = resolve_obs_groups(schema, None)
    assert resolved["actor"].keys == schema.keys
    assert resolved["critic"].keys == schema.keys


def test_resolve_obs_groups_asymmetric():
    schema = _full_schema()
    groups = ObsGroups(actor=("rgb_base_camera",), critic=("state", "rgb_base_camera"))
    resolved = resolve_obs_groups(schema, groups)
    assert resolved["actor"].keys == ("rgb_base_camera",)
    assert set(resolved["critic"].keys) == {"state", "rgb_base_camera"}


def test_resolve_obs_groups_partial_none_defaults_to_full():
    schema = _full_schema()
    groups = ObsGroups(actor=("state",), critic=None)
    resolved = resolve_obs_groups(schema, groups)
    assert resolved["actor"].keys == ("state",)
    assert resolved["critic"].keys == schema.keys


def test_resolve_obs_groups_unknown_key_raises():
    schema = _full_schema()
    groups = ObsGroups(actor=("nonexistent",))
    with pytest.raises(ObservationContractError):
        resolve_obs_groups(schema, groups)
