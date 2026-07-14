from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.buffers.demo_intervention import DemoInterventionMixin
from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin


class _FakeAlgo(DemoInterventionMixin):
    """Minimal host object providing the attributes
    ``_build_prior_data_buffer``/``_sample_train_batch`` rely on."""

    def __init__(self) -> None:
        self._init_prior_data_params()
        self.env = _Env()
        self.nstep = 1
        self.buffer_device = "cpu"
        self.device = "cpu"
        self.replay_buffer = self._build_prior_data_buffer(16)


class _Env:
    single_observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
    single_action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)


def test_init_demo_buffer_sets_empty_growable_buffer_in_offline_slot():
    algo = _FakeAlgo()

    algo.init_demo_buffer(buffer_size=8, demo_data_ratio=0.5)

    assert algo.offline_replay_buffer is not None
    assert algo.offline_data_ratio == 0.5
    assert len(algo.offline_replay_buffer) == 0


def test_add_demo_transition_is_sampled_by_inherited_sample_train_batch():
    algo = _FakeAlgo()
    algo.init_demo_buffer(buffer_size=8, demo_data_ratio=1.0)

    obs = torch.zeros(3)
    action = torch.zeros(2)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    for _ in range(4):
        algo.add_demo_transition(obs, obs, action, reward, done)

    assert len(algo.offline_replay_buffer) == 4
    # demo_data_ratio=1.0 with an empty online replay_buffer: PriorDataReplayMixin
    # (inherited, unmodified) falls back to pure offline-buffer sampling.
    sample = PriorDataReplayMixin._sample_train_batch(algo, batch_size=4)
    assert sample.actions.shape == (4, 2)


def test_rejects_invalid_ratio():
    algo = _FakeAlgo()
    try:
        algo.init_demo_buffer(buffer_size=8, demo_data_ratio=1.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_rejects_overwriting_an_already_populated_offline_replay_buffer():
    algo = _FakeAlgo()
    # Simulates load_offline_replay_buffer(--offline_dataset_path) already
    # having populated this slot -- init_demo_buffer must not silently
    # discard it (regression: it used to overwrite with an empty buffer).
    algo.offline_replay_buffer = algo._build_prior_data_buffer(8)
    algo.offline_data_ratio = 0.3

    try:
        algo.init_demo_buffer(buffer_size=8, demo_data_ratio=0.5)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
