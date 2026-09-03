"""Tests for loading robomimic HDF5 datasets into replay buffers."""

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.buffers import (
    MCTensorReplayBuffer,
    TensorReplayBuffer,
    infer_specs_from_robomimic,
    load_robomimic_dataset_to_replay_buffer,
)
from rl_garden.buffers.robomimic_dataset import ROBOMIMIC_LOW_DIM_OBS_KEYS

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

    assert isinstance(obs_space, spaces.Box)
    assert obs_space.shape == (_OBS_DIM,)
    assert action_space.shape == (7,)
    assert np.all(action_space.low == -1.0)
    assert np.all(action_space.high == 1.0)


def test_load_robomimic_dataset_to_replay_buffer(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
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
    assert sample.obs.shape == (4, _OBS_DIM)
    assert sample.next_obs.shape == (4, _OBS_DIM)
    assert sample.actions.shape == (4, 7)
    # Design decision: dones passed to the buffer are always zero -- the
    # online env never sets terminated=True, so the offline half of a mixed
    # RLPD batch must not cut bootstrap either.
    assert torch.all(sample.dones == 0.0)


def test_load_robomimic_dataset_episode_ends_track_raw_dones(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=2, length=5, action_dim=7)

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
        action_space=spaces.Box(-1.0, 1.0, (7,), dtype=np.float32),
        num_envs=2,
        buffer_size=10,
        gamma=0.9,
        storage_device="cpu",
        sample_device="cpu",
    )

    loaded = load_robomimic_dataset_to_replay_buffer(buffer, path)

    assert loaded == 10
    # Concatenated stream: demo_0's 5 steps (flat index 0..4), then demo_1's
    # 5 steps (flat index 5..9). Loaded in (num_envs=2)-sized chunks -> row 2
    # is [demo_0 step4 (True, its last step), demo_1 step0 (False)]; row 4 is
    # [demo_1 step3 (False), demo_1 step4 (True, its last step)].
    assert buffer._episode_end[2, :].tolist() == [True, False]
    assert buffer._episode_end[4, :].tolist() == [False, True]
    assert buffer._episode_end[0, :].tolist() == [False, False]
    # MC returns must have been computed per-demo (not leaking across the
    # concatenated stream) -- external_mc/externally_valid populated.
    assert buffer._externally_valid[:10].all()


def test_load_robomimic_dataset_num_traj_limits_demos(tmp_path):
    path = tmp_path / "robomimic.hdf5"
    _write_dataset(path, num_demos=3, length=4, action_dim=7)

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
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

    buffer = TensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
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

    buffer = MCTensorReplayBuffer(
        observation_space=spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32),
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


def test_flatten_robomimic_obs_order_matches_constant():
    from rl_garden.buffers.robomimic_dataset import flatten_robomimic_obs

    obs = {key: np.full((dim,), i, dtype=np.float64) for i, (key, dim) in enumerate(_KEY_DIMS.items())}
    flat = flatten_robomimic_obs(obs)
    assert flat.shape == (_OBS_DIM,)
    expected = np.concatenate(
        [obs[key] for key in ROBOMIMIC_LOW_DIM_OBS_KEYS], axis=-1
    )
    assert np.array_equal(flat, expected)
