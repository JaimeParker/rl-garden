"""``RLPDHybrid``'s demo buffer carries the ``grasp_penalty`` column exactly
when ``use_grasp_penalty`` makes the online buffer sample it, so mixed batches
concatenate and the discrete critic's shaped target sees the demo rows too."""

from __future__ import annotations

import pytest
import torch
from test_rlpd_hybrid import DictDummyVecEnv

from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
from rl_garden.buffers.prior_data_replay import PriorDataReplayMixin
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.common.types import GraspPenaltyReplayBufferSample, ReplayBufferSample

BATCH_SIZE = 8
DEMO_PENALTY = -0.05
ONLINE_PENALTY = -0.2


def hybrid_agent(**overrides) -> RLPDHybrid:
    kwargs = {
        "env": DictDummyVecEnv(),
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": BATCH_SIZE,
        "learning_starts": 1,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [8],
        "discrete_hidden_dim": 8,
    }
    kwargs.update(overrides)
    return RLPDHybrid(**kwargs)


def add_demo_transitions(agent: RLPDHybrid, count: int, **extra) -> None:
    obs = {"state": torch.zeros(4)}
    for _ in range(count):
        agent.add_demo_transition(
            obs, obs, torch.zeros(3), torch.tensor(1.0), torch.tensor(False), **extra
        )


def add_online_transitions(agent: RLPDHybrid, steps: int, grasp_penalty: float) -> None:
    num_envs = agent.num_envs
    for _ in range(steps):
        agent.replay_buffer.add(
            {"state": torch.randn(num_envs, 4)},
            {"state": torch.randn(num_envs, 4)},
            torch.zeros(num_envs, 3),
            torch.ones(num_envs),
            torch.zeros(num_envs),
            grasp_penalty=torch.full((num_envs,), grasp_penalty),
        )


def test_demo_buffer_stores_and_samples_grasp_penalty():
    agent = hybrid_agent(use_grasp_penalty=True)

    agent.init_demo_buffer(16, 0.5)

    assert type(agent.offline_replay_buffer) is ReplayBuffer
    assert agent.offline_replay_buffer.store_grasp_penalty is True
    assert agent.offline_replay_buffer.num_envs == 1

    add_demo_transitions(agent, 4, grasp_penalty=torch.tensor(DEMO_PENALTY))
    sample = agent.offline_replay_buffer.sample(4)

    assert type(sample) is GraspPenaltyReplayBufferSample
    torch.testing.assert_close(sample.grasp_penalty, torch.full((4,), DEMO_PENALTY))


def test_demo_transition_without_grasp_penalty_is_stored_as_zero():
    agent = hybrid_agent(use_grasp_penalty=True)
    agent.init_demo_buffer(16, 0.5)

    add_demo_transitions(agent, 2)
    sample = agent.offline_replay_buffer.sample(2)

    torch.testing.assert_close(sample.grasp_penalty, torch.zeros(2))


@pytest.mark.parametrize("utd", [0.5, 2.0], ids=["one_batch_per_step", "high_utd_slices"])
def test_mixed_batch_carries_both_buffers_penalties_and_training_runs(utd):
    agent = hybrid_agent(use_grasp_penalty=True, utd=utd)
    agent.init_demo_buffer(16, 0.5)
    add_demo_transitions(agent, 8, grasp_penalty=torch.tensor(DEMO_PENALTY))
    add_online_transitions(agent, 8, ONLINE_PENALTY)

    batch = agent._sample_train_batch(BATCH_SIZE)

    assert type(batch) is GraspPenaltyReplayBufferSample
    from_demo = torch.isclose(batch.grasp_penalty, torch.tensor(DEMO_PENALTY))
    from_online = torch.isclose(batch.grasp_penalty, torch.tensor(ONLINE_PENALTY))
    assert int(from_demo.sum()) == BATCH_SIZE // 2
    assert int(from_online.sum()) == BATCH_SIZE // 2

    agent.train(2)


def test_demo_buffer_without_grasp_penalty_is_the_plain_mixin_buffer():
    agent = hybrid_agent()

    agent.init_demo_buffer(16, 0.5)

    demo_buffer = agent.offline_replay_buffer
    mixin_buffer = PriorDataReplayMixin._build_prior_data_buffer(agent, 16)
    assert type(demo_buffer) is type(mixin_buffer) is ReplayBuffer
    assert demo_buffer.store_grasp_penalty is False
    assert not hasattr(demo_buffer, "grasp_penalty")
    assert demo_buffer.per_env_buffer_size == mixin_buffer.per_env_buffer_size
    assert demo_buffer.num_envs == mixin_buffer.num_envs == 1

    add_demo_transitions(agent, 2)
    assert type(demo_buffer.sample(2)) is ReplayBufferSample
    with pytest.raises(ValueError, match="store_grasp_penalty=False"):
        add_demo_transitions(agent, 1, grasp_penalty=torch.tensor(DEMO_PENALTY))
