from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import TD3


class DummyDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(32, 40, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def _make_agent(**overrides) -> TD3:
    kwargs = dict(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=2,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        hidden_dim=16,
        feature_dim=8,
        image_keys=("rgb",),
        proprio_latent_dim=4,
        image_augmentation="none",
    )
    kwargs.update(overrides)
    return TD3(**kwargs)


def _add_transition(agent: TD3, reward: float) -> None:
    obs = {
        "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
        "state": torch.randn(1, 4),
    }
    next_obs = {
        "rgb": torch.randint(0, 256, (1, 32, 40, 3), dtype=torch.uint8),
        "state": torch.randn(1, 4),
    }
    agent.replay_buffer.add(
        obs=obs,
        next_obs=next_obs,
        action=torch.randn(1, 2).clamp(-1, 1),
        reward=torch.full((1,), reward),
        done=torch.zeros(1, dtype=torch.bool),
        episode_end=torch.zeros(1, dtype=torch.bool),
    )


def test_td3_one_update_smoke():
    agent = _make_agent(nstep=3, gamma=0.9)
    for step in range(5):
        _add_transition(agent, float(step))

    metrics = agent.train(1, compute_info=True)

    assert set(metrics) >= {"critic_loss", "target_q", "predicted_q"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_td3_target_action_noise_is_fixed_and_decoupled_from_schedule():
    agent = _make_agent(
        target_noise_std=0.2,
        target_noise_clip=0.5,
        stddev_schedule="linear(1.0,0.1,100)",
        stddev_clip=0.3,
    )

    std0, clip0 = agent._target_action_noise()
    agent._global_step = 1000
    std1, clip1 = agent._target_action_noise()

    assert std0 == std1 == 0.2
    assert clip0 == clip1 == 0.5
    # Contrast with the exploration schedule, which decays with _global_step
    # (DDPG's default _target_action_noise reuses it; TD3's does not).
    assert agent._current_stddev() < 1.0


def test_td3_actor_q_value_uses_first_critic_only():
    agent = _make_agent()
    q_actor_all = torch.tensor([[[5.0], [6.0]], [[1.0], [2.0]]])  # (2, B=2, 1); q1 is NOT the min

    reduced = agent._actor_q_value(q_actor_all)

    assert torch.equal(reduced, q_actor_all[0])
    assert not torch.equal(reduced, q_actor_all.min(dim=0).values)


def test_td3_delays_actor_and_target_update():
    agent = _make_agent(nstep=1, gamma=0.9, policy_freq=2, tau=0.9)
    for step in range(5):
        _add_transition(agent, float(step))

    actor_before = [p.clone() for p in agent.policy.actor_parameters()]
    target_before = [p.clone() for p in agent.policy.critic_target.parameters()]

    # First gradient step: _global_update becomes 1, 1 % 2 != 0 -> actor/target skipped.
    metrics = agent.train(1, compute_info=True)
    assert "actor_loss" not in metrics
    for before, after in zip(actor_before, agent.policy.actor_parameters()):
        assert torch.equal(before, after)
    for before, after in zip(target_before, agent.policy.critic_target.parameters()):
        assert torch.equal(before, after)

    # Second gradient step: _global_update becomes 2, 2 % 2 == 0 -> actor/target update.
    metrics = agent.train(1, compute_info=True)
    assert "actor_loss" in metrics
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, agent.policy.actor_parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, agent.policy.critic_target.parameters())
    )


def test_td3_checkpoint_roundtrip(tmp_path):
    agent = _make_agent(nstep=1, gamma=0.9)
    for step in range(5):
        _add_transition(agent, float(step))
    agent.train(2)

    path = tmp_path / "td3.pt"
    agent.save(path)

    loaded = _make_agent(nstep=1, gamma=0.9)
    loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_td3_rejects_bad_construction_like_ddpg():
    with pytest.raises(ValueError):
        TD3(
            env=DummyDictVecEnv(),
            device="cpu",
            buffer_device="cpu",
            replay_lazy_next_obs=True,
            save_replay_buffer=True,
        )
