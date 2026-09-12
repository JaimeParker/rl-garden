from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.offline_sac import OfflineSAC
from rl_garden.buffers import ReplayBuffer


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict({"state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)}),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _dict_image_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "rgb_cam": spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> OfflineSAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
    )
    defaults.update(kwargs)
    return OfflineSAC(**defaults)


def test_uses_dict_replay_buffer_for_bare_box_env():
    # A bare Box env is boundary-normalized to Dict({"state": Box}) by
    # BaseAlgorithm.__init__ -- OfflineSAC's replay buffer must be the Dict
    # variant regardless.
    agent = _make_agent()
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.observation_encoders.schema.keys == ("state",)


def test_accepts_state_only_dict_observation_space():
    agent = OfflineSAC(env=_dict_env(), buffer_device="cpu", device="cpu")
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.observation_encoders.schema.keys == ("state",)


def test_accepts_dict_observation_space_with_images():
    # OfflineSAC (via SACCore) supports vision through the shared encoder
    # mixin -- a Dict env with an image key must construct without error.
    agent = OfflineSAC(env=_dict_image_env(), buffer_device="cpu", device="cpu")
    assert agent.observation_encoders.schema.has_images
