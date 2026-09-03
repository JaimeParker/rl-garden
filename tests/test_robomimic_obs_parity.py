"""Regression test for the single highest-value correctness property of the
robomimic backend: the online env's flattened observation and the offline
dataset loader's flattened observation must use the exact same key set and
order. If they ever diverged, RLPD's offline/online mixed batches would be
silently misaligned -- trains, loss goes down, policy never actually
improves, with no other test naturally catching it.
"""
from __future__ import annotations

import h5py
import numpy as np

from rl_garden.buffers.robomimic_dataset import (
    ROBOMIMIC_LOW_DIM_OBS_KEYS,
    flatten_robomimic_obs,
    infer_specs_from_robomimic,
)
from rl_garden.envs.robomimic.config import RobomimicEnvConfig
from rl_garden.envs.robomimic.env import _RobomimicGymV21Adapter

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


class _FixedObsEnv:
    """A fake robosuite env whose obs dict always returns the same
    per-key values, plus one extra key not present in the offline dataset
    (matching robomimic's real force-activated-observable behavior)."""

    def __init__(self, per_key_values: dict[str, np.ndarray]) -> None:
        self.action_dimension = 7
        self._values = per_key_values

    def _obs(self):
        obs = dict(self._values)
        obs["robot0_eef_vel_lin"] = np.zeros(6, dtype=np.float64)
        return obs

    def reset(self):
        return self._obs()

    def step(self, action):
        return self._obs(), 0.0, False, {"is_success": {"task": False}}

    # Deliberately no close() here -- EnvRobosuite itself exposes none;
    # rl-garden's adapter must reach through to self.env.close() instead.
    # (Not exercised by this test, which never closes the env.)


def test_env_and_dataset_flatten_agree_on_key_order_and_values(tmp_path):
    rng = np.random.default_rng(0)
    per_key_values = {
        key: rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float64)
        for key, dim in _KEY_DIMS.items()
    }

    # Online side: flatten via the actual env adapter.
    env = _RobomimicGymV21Adapter(_FixedObsEnv(per_key_values))
    online_obs, _ = env.reset()

    # Offline side: flatten the same per-key values as read out of a real
    # hdf5 file (through the dataset loader's own code path).
    path = tmp_path / "robomimic.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float64))
        demo.create_dataset("rewards", data=np.zeros(1, dtype=np.float64))
        demo.create_dataset("dones", data=np.array([1], dtype=np.int64))
        obs_group = demo.create_group("obs")
        next_obs_group = demo.create_group("next_obs")
        for key, value in per_key_values.items():
            obs_group.create_dataset(key, data=value.reshape(1, -1))
            next_obs_group.create_dataset(key, data=value.reshape(1, -1))

    with h5py.File(path, "r") as f:
        offline_obs = flatten_robomimic_obs(
            {key: f["data"]["demo_0"]["obs"][key][0] for key in ROBOMIMIC_LOW_DIM_OBS_KEYS}
        )

    assert online_obs.shape == (_OBS_DIM,)
    assert np.allclose(online_obs.astype(np.float64), offline_obs, atol=1e-6)

    obs_space, _ = infer_specs_from_robomimic(path)
    assert obs_space.shape == (_OBS_DIM,)
