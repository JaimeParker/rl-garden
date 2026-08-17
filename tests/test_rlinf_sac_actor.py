"""RLinf-independent tests for rl_garden.integrations.rlinf.sac_actor.

RLinf isn't required to run these: they cover the algorithm-dispatch table,
the trajectory-batch conversion (including the terminations-vs-dones
mapping), the replay-buffer shim's fail-loud behavior, and -- the key
regression this design exists to prevent -- that swapping
``self._algo.replay_buffer`` on an RLPD instance still routes through
``PriorDataReplayMixin``'s offline/online mixing rather than silently
degrading to plain-SAC behavior. See
``rl_garden/integrations/rlinf/sac_actor.py``'s module docstring and
``docs/design/rlinf-integration.md``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import RLPD, SAC
from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.algorithms.sac_core import SACCore
from rl_garden.integrations.rlinf.sac_actor import (
    _ALGORITHMS,
    _TrajectoryReplayBufferShim,
    resolve_algorithm,
    trajectory_batch_to_sample,
)


def test_algorithm_dispatch_covers_saccore_contract():
    assert _ALGORITHMS == {"sac": SAC, "rlpd": RLPD}


@pytest.mark.parametrize("name", ["sac", "rlpd"])
def test_resolve_algorithm_returns_expected_class(name):
    assert resolve_algorithm(name) is _ALGORITHMS[name]


def test_resolve_algorithm_rejects_td3():
    with pytest.raises(ValueError, match="DDPG contract"):
        resolve_algorithm("td3")


def test_resolve_algorithm_rejects_rlpd_hybrid():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_algorithm("rlpd_hybrid")


def test_trajectory_batch_to_sample_uses_terminations_not_dones():
    # dones = terminations | truncations (env_worker.py:486) -- must not be
    # used for rl-garden's bootstrap-suppression `dones` field, which
    # should only fire on true termination, not truncation.
    batch = {
        "curr_obs": {"states": torch.zeros(4, 3)},
        "next_obs": {"states": torch.ones(4, 3)},
        "actions": torch.full((4, 1, 2), 0.5),
        "rewards": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "terminations": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
        "truncations": torch.tensor([[0.0], [1.0], [0.0], [0.0]]),
        "dones": torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
    }
    sample = trajectory_batch_to_sample(batch, torch.device("cpu"))
    assert sample.obs.shape == (4, 3)
    assert sample.actions.shape == (4, 2)
    assert sample.rewards.shape == (4,)
    assert torch.equal(sample.dones, torch.tensor([0.0, 0.0, 1.0, 0.0]))


class _FakeRawBuffer:
    """Stand-in for RLinf's TrajectoryReplayBuffer, deterministic samples."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 2, size: int = 100) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._size = size

    def sample(self, num_chunks: int) -> dict:
        num_chunks = min(num_chunks, self._size)
        if num_chunks == 0:
            return {}
        return {
            "curr_obs": {"states": torch.zeros(num_chunks, self.obs_dim)},
            "next_obs": {"states": torch.ones(num_chunks, self.obs_dim)},
            "actions": torch.zeros(num_chunks, 1, self.action_dim),
            "rewards": torch.ones(num_chunks, 1),
            "terminations": torch.zeros(num_chunks, 1),
            "truncations": torch.zeros(num_chunks, 1),
            "dones": torch.zeros(num_chunks, 1),
        }

    def __len__(self) -> int:
        return self._size


def test_shim_sample_returns_replay_buffer_sample():
    shim = _TrajectoryReplayBufferShim(_FakeRawBuffer(), torch.device("cpu"))
    sample = shim.sample(8)
    assert sample.obs.shape == (8, 4)
    assert sample.actions.shape == (8, 2)
    assert len(shim) == 100


def test_shim_raises_on_empty_batch():
    class _EmptyRaw:
        def sample(self, n):
            return {}

        def __len__(self):
            return 0

    shim = _TrajectoryReplayBufferShim(_EmptyRaw(), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="empty batch"):
        shim.sample(8)


def test_shim_raises_on_short_batch():
    shim = _TrajectoryReplayBufferShim(_FakeRawBuffer(size=4), torch.device("cpu"))
    with pytest.raises(RuntimeError, match="returned 4 transitions"):
        shim.sample(8)


class _DummyVecEnv:
    """Minimal env-like object for construction, mirroring OfflineEnvSpec's
    shape (matches the already-established finding that SAC/RLPD
    construction needs no live env -- see tests/test_rlpd.py's own
    DummyVecEnv for precedent)."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)


def _rlpd_agent(**overrides) -> RLPD:
    kwargs = dict(
        env=_DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
    )
    kwargs.update(overrides)
    return RLPD(**kwargs)


def test_replay_buffer_swap_preserves_rlpd_mixing():
    """The key regression this design exists to prevent.

    Swapping self._algo.replay_buffer to the RLinf-backed shim must not
    disturb RLPD's own _sample_train_batch (PriorDataReplayMixin's
    offline/online mixing) -- confirming the injection point is the buffer
    object, not a monkey-patched method (which would silently make RLPD
    train as plain SAC with zero offline data; see sac_actor.py's
    _TrajectoryReplayBufferShim docstring).
    """
    agent = _rlpd_agent()

    # RLPD must still resolve _sample_train_batch to PriorDataReplayMixin's
    # override, never SACCore's plain one -- confirms nothing at the
    # instance level (like a Phase-1-style monkey-patch) shadows it.
    assert RLPD._sample_train_batch is PriorDataReplayMixin._sample_train_batch
    assert RLPD._sample_train_batch is not SACCore._sample_train_batch

    # Swap the online buffer object for the RLinf-backed shim (the actual
    # injection point sac_actor.py's init_worker uses).
    agent.replay_buffer = _TrajectoryReplayBufferShim(
        _FakeRawBuffer(obs_dim=4, action_dim=2, size=1000), torch.device("cpu")
    )

    # Populate a real offline buffer and enable mixing, exactly like
    # tests/test_rlpd.py's own test_rlpd_offline_mixing_batch_shape.
    agent.offline_replay_buffer = agent._build_prior_data_buffer(32)
    for _ in range(4):
        agent.offline_replay_buffer.add(
            torch.full((1, 4), 9.0),
            torch.full((1, 4), 9.0),
            torch.full((1, 2), 9.0),
            torch.full((1,), 9.0),
            torch.zeros(1),
        )
    agent.offline_data_ratio = 0.5

    batch = agent._sample_train_batch(8)
    assert batch.obs.shape == (8, 4)
    assert batch.actions.shape == (8, 2)
    assert batch.rewards.shape == (8,)

    # Mixing actually happened: some rows come from the offline buffer
    # (marked 9.0) and some from the shim's curr_obs (marked 0.0 per
    # _FakeRawBuffer), not exclusively one or the other -- proves the
    # shim-backed online buffer and the offline buffer were both actually
    # sampled from.
    obs_values = set(batch.obs.flatten().tolist())
    assert 9.0 in obs_values, "offline buffer contributed no rows -- mixing broke"
    assert 0.0 in obs_values, "shim-backed online buffer contributed no rows -- mixing broke"
