from __future__ import annotations

import numpy as np
import pytest
import torch
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
    defaults = {
        "env": _state_env(),
        "buffer_size": 1000,
        "buffer_device": "cpu",
        "batch_size": 32,
        "device": "cpu",
    }
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


def _fill(agent: OfflineSAC, steps: int = 64) -> None:
    env = agent.env
    obs_shape = env.single_observation_space["state"].shape
    for _ in range(steps):
        state = torch.randn(env.num_envs, *obs_shape)
        next_state = torch.randn_like(state)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add({"state": state}, {"state": next_state}, actions, rewards, dones)


def test_backup_entropy_defaults_to_true_without_manual_assignment():
    agent = _make_agent()
    assert agent.backup_entropy is True
    assert agent._backup_entropy_enabled() is True


def test_plain_offline_sac_trains_one_step():
    agent = _make_agent(net_arch=[16])
    _fill(agent)
    metrics = agent.train(1, compute_info=True)
    assert np.isfinite(metrics["td_loss"])


def test_backup_entropy_false_changes_the_target():
    agent = _make_agent(net_arch=[16], backup_entropy=False)
    assert agent.backup_entropy is False
    _fill(agent)
    data = agent.replay_buffer.sample(agent.batch_size)

    next_action = torch.zeros(agent.batch_size, *agent.env.single_action_space.shape)
    next_log_prob = torch.full((agent.batch_size, 1), 2.0)
    next_features = agent.policy.extract_features(data.next_obs, stop_gradient=True)
    agent._target_action_log_prob = lambda data: (next_action, next_log_prob, next_features)

    target_without_entropy = agent._target_q(data)
    agent.backup_entropy = True
    target_with_entropy = agent._target_q(data)

    alpha = agent._current_alpha().detach()
    expected_diff = -agent._target_discounts(data) * alpha * next_log_prob
    torch.testing.assert_close(target_with_entropy - target_without_entropy, expected_diff)
    assert not torch.allclose(target_with_entropy, target_without_entropy)


@pytest.mark.parametrize("backup_entropy", [True, False])
def test_checkpoint_metadata_records_backup_entropy(backup_entropy):
    agent = _make_agent(backup_entropy=backup_entropy)
    assert agent._checkpoint_metadata()["backup_entropy"] is backup_entropy
