"""Tests for loading robomimic HDF5 datasets into replay buffers."""

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    DatasetRequest,
    MCReplayBuffer,
    ReplayBuffer,
    infer_specs_from_robomimic,
    load_dataset,
    load_robomimic_dataset_to_replay_buffer,
)
from rl_garden.buffers.chunked_replay_buffer import ChunkedReplayBuffer
from rl_garden.buffers.robomimic_dataset import ROBOMIMIC_LOW_DIM_OBS_KEYS
from rl_garden.common.env_args import RobomimicConfig

_KEY_DIMS = {
    "object": 10,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "robot0_gripper_qvel": 2,
    "robot0_joint_pos": 7,
    "robot0_joint_pos_cos": 7,
    "robot0_joint_pos_sin": 7,
    "robot0_joint_vel": 7,
}
_OBS_DIM = sum(_KEY_DIMS.values())


def _write_demo(group, *, length, action_dim, dones, reward_value, obs_value):
    group.create_dataset(
        "actions", data=np.full((length, action_dim), 0.5, dtype=np.float64)
    )
    group.create_dataset("rewards", data=np.full(length, reward_value, dtype=np.float64))
    group.create_dataset("dones", data=np.asarray(dones, dtype=np.int64))
    group.create_dataset("states", data=np.zeros((length, 5), dtype=np.float64))
    obs = group.create_group("obs")
    next_obs = group.create_group("next_obs")
    for key, dim in _KEY_DIMS.items():
        obs.create_dataset(key, data=np.full((length, dim), obs_value, dtype=np.float64))
        next_obs.create_dataset(key, data=np.full((length, dim), obs_value, dtype=np.float64))
    return group


def _write_dataset(path, *, num_demos=2, length=5, action_dim=7, env_kwargs=None):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        import json

        data.attrs["env_args"] = json.dumps(
            {"env_name": "Lift", "env_kwargs": env_kwargs or {}}
        )
        for i in range(num_demos):
            dones = [0] * (length - 1) + [1]
            demo = data.create_group(f"demo_{i}")
            _write_demo(
                demo,
                length=length,
                action_dim=action_dim,
                dones=dones,
                reward_value=float(i),
                obs_value=float(i),
            )
        mask = f.create_group("mask")
        mask.create_dataset("train", data=np.arange(num_demos))


def test_infer_specs_from_robomimic(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)

    obs_space, action_space = infer_specs_from_robomimic(path)

    assert isinstance(obs_space, spaces.Dict)
    assert set(obs_space.spaces) == {"state"}
    assert obs_space["state"].shape == (_OBS_DIM,)
    assert action_space.shape == (7,)
    assert np.all(action_space.low == -1.0)
    assert np.all(action_space.high == 1.0)


def test_load_robomimic_dataset_to_replay_buffer(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)

    buffer = ReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_robomimic_dataset_to_replay_buffer(buffer, path)

    assert loaded == 10
    assert len(buffer) == 10
    sample = buffer.sample(4)
    assert sample.obs["state"].shape == (4, _OBS_DIM)
    assert sample.next_obs["state"].shape == (4, _OBS_DIM)
    assert sample.actions.shape == (4, 7)
    # Design decision: dones passed to the buffer are always zero -- the
    # online env never sets terminated=True, so the offline half of a mixed
    # RLPD batch must not cut bootstrap either.
    assert torch.all(sample.dones == 0.0)


def test_load_robomimic_dataset_episode_ends_track_raw_dones(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_robomimic_dataset_to_replay_buffer(buffer, path)

    assert loaded == 10
    # One contiguous stretch of the concatenated stream per env: demo_0's 5
    # steps in env 0, demo_1's in env 1, each ending at its last step.
    assert buffer._episode_end[:5, :].tolist() == [[False, False]] * 4 + [[True, True]]
    assert buffer.obs["state"][:5, :, 0].tolist() == [[0.0, 1.0]] * 5
    # MC returns must have been computed per-demo (not leaking across the
    # concatenated stream) -- external_mc/externally_valid populated.
    assert buffer._externally_valid[:10].all()


def test_load_robomimic_dataset_num_traj_limits_demos(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=3, length=4, action_dim=7)

    buffer = ReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=1,
        buffer_size=20,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_robomimic_dataset_to_replay_buffer(buffer, path, num_traj=2)

    assert loaded == 8  # 2 demos * length 4, not all 3


def test_load_robomimic_dataset_missing_obs_key_raises(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float64))
        demo.create_dataset("rewards", data=np.zeros(3, dtype=np.float64))
        demo.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.int64))
        obs = demo.create_group("obs")
        next_obs = demo.create_group("next_obs")
        # Deliberately omit "robot0_gripper_qvel" -- one of
        # ROBOMIMIC_LOW_DIM_OBS_KEYS.
        for key, dim in _KEY_DIMS.items():
            if key == "robot0_gripper_qvel":
                continue
            obs.create_dataset(key, data=np.zeros((3, dim), dtype=np.float64))
            next_obs.create_dataset(key, data=np.zeros((3, dim), dtype=np.float64))

    buffer = ReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        storage_device="cpu",
        sample_device="cpu",
    )

    with pytest.raises(KeyError, match="robot0_gripper_qvel"):
        load_robomimic_dataset_to_replay_buffer(buffer, path)


def test_load_robomimic_dataset_infers_success_when_sparse_reward_mc_enabled(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        _write_demo(
            demo,
            length=3,
            action_dim=7,
            dones=[0, 0, 1],
            reward_value=0.0,
            obs_value=0.0,
        )
        # rewrite rewards so the last step is a success under threshold 0.5
        del demo["rewards"]
        demo.create_dataset("rewards", data=np.array([0.0, 0.0, 1.0], dtype=np.float64))

    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=1,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=True,
        success_threshold=0.5,
    )

    with pytest.warns(RuntimeWarning, match="inferring success"):
        loaded = load_robomimic_dataset_to_replay_buffer(buffer, path)

    assert loaded == 3


def _load_one_demo(tmp_path, demo_rewards, config, *, sparse_reward_mc=False, **request):
    """Load one 4-step demo with raw rewards ``demo_rewards`` (end marker on the last
    step) through the registered backend into an MC buffer with gamma 0.5."""
    path = tmp_path / "robomimic.hdf5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data").create_group("demo_0")
        _write_demo(
            demo, length=4, action_dim=7, dones=[0, 0, 0, 1], reward_value=0.0, obs_value=0.0
        )
        demo["rewards"][:] = np.asarray(demo_rewards, dtype=np.float64)
    buffer = MCReplayBuffer(
        observation_space=spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=1,
        buffer_size=4,
        gamma=0.5,
        storage_device="cpu",
        sample_device="cpu",
        sparse_reward_mc=sparse_reward_mc,
    )
    req = DatasetRequest(path=str(path), backend_config=config, **request)
    assert load_dataset(buffer, req, backend_name="robomimic") == 4
    return buffer


def test_terminate_on_success_marks_success_steps_done_and_stops_mc_returns_there(tmp_path):
    buffer = _load_one_demo(
        tmp_path, [0.0, 1.0, 0.0, 1.0], RobomimicConfig(terminate_on_success=True)
    )

    assert buffer.dones[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert buffer._episode_end[:, 0].tolist() == [False, True, False, True]
    assert buffer._external_mc[:, 0].tolist() == [0.5, 1.0, 0.5, 1.0]


def test_terminate_on_success_leaves_a_failed_demo_without_dones(tmp_path):
    buffer = _load_one_demo(
        tmp_path, [0.0, 0.0, 0.0, 0.0], RobomimicConfig(terminate_on_success=True)
    )

    assert buffer.dones[:, 0].tolist() == [0.0] * 4
    assert buffer._episode_end[:, 0].tolist() == [False, False, False, True]


def test_terminate_on_success_uses_the_raw_reward_and_ignores_shaping_below_success(tmp_path):
    # 0.6 stands for a shaped reward of an unsuccessful step: robosuite only
    # reaches 1.0 on task success.
    buffer = _load_one_demo(
        tmp_path,
        [0.0, 0.6, 0.0, 1.0],
        RobomimicConfig(terminate_on_success=True),
        reward_scale=10.0,
        reward_bias=-5.0,
    )

    assert buffer.rewards[:, 0].tolist() == [-5.0, 1.0, -5.0, 5.0]
    assert buffer.dones[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.filterwarnings("ignore:sparse_reward_mc=True")
def test_terminate_on_success_stops_sparse_reward_mc_returns_at_success(tmp_path):
    buffer = _load_one_demo(
        tmp_path,
        [0.0, 1.0, 0.0, 1.0],
        RobomimicConfig(terminate_on_success=True),
        sparse_reward_mc=True,
    )

    assert buffer._build_mc_table()[:, 0].tolist() == [0.5, 1.0, 0.5, 1.0]


@pytest.mark.parametrize(
    "config", [None, RobomimicConfig()], ids=["no-backend-config", "default-backend-config"]
)
def test_without_terminate_on_success_dones_stay_zero(tmp_path, config):
    buffer = _load_one_demo(tmp_path, [0.0, 1.0, 0.0, 1.0], config)

    assert buffer.dones[:, 0].tolist() == [0.0] * 4
    assert buffer._episode_end[:, 0].tolist() == [False, False, False, True]
    assert buffer._external_mc[:, 0].tolist() == [0.625, 1.25, 0.5, 1.0]


def test_flatten_robomimic_obs_order_matches_constant():
    from rl_garden.buffers.robomimic_dataset import flatten_robomimic_obs

    obs = {key: np.full((dim,), i, dtype=np.float64) for i, (key, dim) in enumerate(_KEY_DIMS.items())}
    flat = flatten_robomimic_obs(obs)
    assert flat.shape == (_OBS_DIM,)
    expected = np.concatenate(
        [obs[key] for key in ROBOMIMIC_LOW_DIM_OBS_KEYS], axis=-1
    )
    assert np.array_equal(flat, expected)


def test_every_demo_ends_where_it_ends_even_without_dones(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)
    with h5py.File(path, "a") as f:
        for i in range(2):
            f[f"data/demo_{i}/dones"][...] = 0
    buffer = ChunkedReplayBuffer(
        spaces.Dict({"state": spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)}),
        spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=1, buffer_size=10, horizon_length=3, gamma=0.9, storage_device="cpu", sample_device="cpu",
    )
    load_robomimic_dataset_to_replay_buffer(buffer, path)

    assert buffer.episode_ends[:10, 0].tolist() == [False] * 4 + [True] + [False] * 4 + [True]
    assert not buffer.dones.any()
    sample = buffer.sample(200)
    demo = sample.obs["state"][:, 0]
    assert torch.equal(sample.next_obs["state"][:, 0], demo)
    starts_at_last_step = sample.valid.sum(1) == 1
    assert starts_at_last_step.any()
    assert torch.allclose(sample.discounts[starts_at_last_step], torch.tensor(0.9))
