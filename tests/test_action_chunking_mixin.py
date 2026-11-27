from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import ACFQL, ACRLPD, QAM, QGF
from rl_garden.algorithms.acfql import ACFQLCore
from rl_garden.algorithms.acrlpd import ACRLPDCore
from rl_garden.algorithms.chunked_rollout import ActionChunkingMixin, ChunkedRolloutMixin
from rl_garden.algorithms.qam import QAMCore
from rl_garden.algorithms.qgf import QGFCore
from rl_garden.common.types import ChunkedReplayBufferSample


class _BaseHarness:
    def _sample_train_batch(self, batch_size):
        self.requested_batch_size = batch_size
        return self.batch

    def _checkpoint_metadata(self):
        return {"parent": "preserved"}


class _Harness(ActionChunkingMixin, _BaseHarness):
    def __init__(self, horizon_length=3):
        self._init_action_chunking(horizon_length)
        self.env = SimpleNamespace(
            single_action_space=spaces.Box(
                low=np.array([-2.0, 0.5], dtype=np.float32),
                high=np.array([0.25, 4.0], dtype=np.float32),
            ),
            single_observation_space=spaces.Dict(
                {"state": spaces.Box(-1.0, 1.0, (2,), np.float32)}
            ),
        )
        self.num_envs = 2
        self.buffer_size = 24
        self.gamma = 0.97
        self.buffer_device = "cpu"
        self.device = torch.device("cpu")


def test_tiles_asymmetric_action_bounds():
    action_space = _Harness()._policy_action_space()
    np.testing.assert_array_equal(action_space.low, [-2.0, 0.5] * 3)
    np.testing.assert_array_equal(action_space.high, [0.25, 4.0] * 3)
    assert action_space.shape == (6,)
    assert action_space.dtype == np.float32


@pytest.mark.parametrize("horizon_length", [0, -1])
def test_rejects_nonpositive_horizon(horizon_length):
    with pytest.raises(ValueError) as exc:
        _Harness(horizon_length)
    assert str(exc.value) == f"horizon_length must be >= 1, got {horizon_length}"


def test_buffer_factory_preserves_all_arguments():
    host = _Harness()
    with patch("rl_garden.algorithms.chunked_rollout.ChunkedReplayBuffer") as factory:
        assert host._chunked_replay_buffer(num_envs=1, buffer_size=12) is factory.return_value
        factory.assert_called_once_with(
            observation_space=host.env.single_observation_space,
            action_space=host.env.single_action_space,
            num_envs=1,
            buffer_size=12,
            horizon_length=3,
            gamma=host.gamma,
            storage_device=host.buffer_device,
            sample_device=host.device,
        )
        factory.reset_mock()
        assert host._build_replay_buffer() is factory.return_value
        assert factory.call_args.kwargs["num_envs"] == host.num_envs
        assert factory.call_args.kwargs["buffer_size"] == host.buffer_size


def test_batch_flatten_preserves_every_other_field():
    host = _Harness()
    host.batch = ChunkedReplayBufferSample(
        obs={"state": torch.zeros(4, 2)},
        next_obs={"state": torch.ones(4, 2)},
        actions=torch.arange(24).reshape(4, 3, 2),
        rewards=torch.arange(4),
        dones=torch.zeros(4, dtype=torch.bool),
        discounts=torch.full((4,), 0.97),
        valid=torch.ones(4, 3, dtype=torch.bool),
    )
    result = host._sample_train_batch(4)
    assert host.requested_batch_size == 4
    assert type(result) is type(host.batch)
    assert result.actions.shape == (4, 6)
    assert torch.equal(result.actions, host.batch.actions.reshape(4, 6))
    assert host.batch.actions.shape == (4, 3, 2)
    for field in dataclasses.fields(host.batch):
        if field.name != "actions":
            assert getattr(result, field.name) is getattr(host.batch, field.name)


def test_metadata_extends_parent():
    assert _Harness()._checkpoint_metadata() == {"parent": "preserved", "horizon_length": 3}


@pytest.mark.parametrize("algorithm,core", [
    (ACRLPD, ACRLPDCore), (ACFQL, ACFQLCore), (QAM, QAMCore), (QGF, QGFCore),
])
def test_real_hosts_resolve_shared_plumbing_without_construction(algorithm, core):
    for method in ("_policy_action_space", "_build_replay_buffer", "_sample_train_batch"):
        owner = next(cls for cls in algorithm.__mro__ if method in cls.__dict__)
        assert owner is ActionChunkingMixin
    assert "_policy_action_space" not in core.__dict__
    assert "_build_replay_buffer" not in core.__dict__
    if algorithm in (ACRLPD, ACFQL):
        assert algorithm.__bases__[0] is ChunkedRolloutMixin
        owner = next(cls for cls in algorithm.__mro__ if "_sample_action_chunk" in cls.__dict__)
        assert owner is ChunkedRolloutMixin
    else:
        assert algorithm.__bases__[0] is ActionChunkingMixin
