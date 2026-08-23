from __future__ import annotations

import io
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid


class _ConstQ(nn.Module):
    """Ignores its input, returns a fixed Q-value tensor -- used to make
    _train_discrete_critic's next_q deterministic and independent of any
    real network's random init or prior optimizer steps."""

    def __init__(self, value: torch.Tensor, requires_grad: bool = False) -> None:
        super().__init__()
        self.value = nn.Parameter(value, requires_grad=requires_grad)

    def forward(self, features):
        del features
        return self.value


class DummyVecEnv:
    """3D action space: 2 continuous ee dims + 1 discrete gripper dim."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 3)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 3)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, 4), {}

    def step(self, actions):
        obs = torch.randn(self.num_envs, 4)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _agent(**overrides) -> RLPDHybrid:
    kwargs = dict(
        env=DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
    )
    kwargs.update(overrides)
    return RLPDHybrid(**kwargs)


def test_learns_without_crashing_and_updates_discrete_critic_independently():
    agent = _agent()
    initial_discrete_params = [p.clone() for p in agent.policy.discrete_critic.parameters()]

    agent.learn(total_timesteps=16)

    assert agent._global_step == 16
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_discrete_params, agent.policy.discrete_critic.parameters())
    ), "discrete_critic params never changed"


def test_predict_returns_concatenated_continuous_and_discrete_action():
    agent = _agent()
    action = agent.policy.predict(torch.zeros(2, 4), deterministic=True)
    assert action.shape == (2, 3)


def test_learns_with_demo_buffer_mixed_in():
    agent = _agent()
    agent.init_demo_buffer(buffer_size=32, demo_data_ratio=0.5)

    obs = torch.zeros(4)
    action = torch.zeros(3)
    reward = torch.tensor(1.0)
    done = torch.tensor(False)
    for _ in range(8):
        agent.add_demo_transition(obs, obs, action, reward, done)

    agent.learn(total_timesteps=16)

    assert agent._global_step == 16
    assert len(agent.offline_replay_buffer) == 8


class DictDummyVecEnv:
    """Dict obs space (required for use_grasp_penalty), same 3D hybrid
    continuous+discrete action space as DummyVecEnv."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, 3)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, 3)),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None):
        del seed
        return {"state": torch.zeros(self.num_envs, 4)}, {}

    def step(self, actions):
        obs = {"state": torch.randn(self.num_envs, 4)}
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        # gripper action is the last dim; flip parity of its sign each step
        # to guarantee some nonzero grasp_penalty during a short rollout.
        info = {"grasp_penalty": torch.full((self.num_envs,), -0.05)}
        return obs, rewards, terminations, truncations, info

    def close(self) -> None:
        return None


def test_use_grasp_penalty_requires_dict_obs_space():
    with pytest.raises(ValueError, match="Dict observation space"):
        _agent(use_grasp_penalty=True)  # DummyVecEnv is Box, not Dict


def test_use_grasp_penalty_learns_without_crashing_at_high_utd():
    # utd > 1 specifically: this is the code path (SACCore.train_high_utd ->
    # _slice_batch -> type(batch)(**kwargs)) that would TypeError if
    # grasp_penalty weren't added to _extra_batch_slice_keys, since
    # GraspPenaltyReplayBufferSample has no default for that field -- a
    # utd=1 test would pass even with that wiring broken.
    agent = RLPDHybrid(
        env=DictDummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
        use_grasp_penalty=True,
        utd=2.0,
    )
    agent.learn(total_timesteps=16)
    assert agent._global_step == 16


def test_grasp_penalty_flag_controls_whether_target_is_shaped():
    """Drives the real _train_discrete_critic() (not a standalone tensor
    expression) with a fake batch and captures the exact `target` tensor
    F.mse_loss is called with, for use_grasp_penalty=True vs False.
    discrete_critic/discrete_target_critic are stubbed to fixed,
    non-parametric outputs so next_q is bit-identical and deterministic
    across both branches -- isolating the grasp_penalty formula itself from
    network randomness or from one call's optimizer step affecting the
    next."""
    import rl_garden.algorithms.rlpd_hybrid as rlpd_hybrid_module

    batch_size = 4
    rewards = torch.full((batch_size,), 1.0)
    penalty = torch.full((batch_size,), -0.05)

    def fake_batch():
        return types.SimpleNamespace(
            obs={"state": torch.zeros(batch_size, 4)},
            next_obs={"state": torch.zeros(batch_size, 4)},
            actions=torch.zeros(batch_size, 3),
            rewards=rewards,
            dones=torch.zeros(batch_size),
            grasp_penalty=penalty,
        )

    captured_targets = []
    real_mse_loss = rlpd_hybrid_module.F.mse_loss

    def spying_mse_loss(q_pred, target):
        captured_targets.append(target.clone())
        return real_mse_loss(q_pred, target)

    def _make(use_grasp_penalty):
        agent = RLPDHybrid(
            env=DictDummyVecEnv(), device="cpu", buffer_device="cpu", buffer_size=64,
            batch_size=8, learning_starts=1, training_freq=4, eval_freq=0, log_freq=0,
            net_arch=[8], discrete_hidden_dim=8, use_grasp_penalty=use_grasp_penalty,
        )
        agent.gamma = 1.0
        agent._sample_train_batch = lambda n: fake_batch()
        agent.policy.discrete_critic = _ConstQ(torch.zeros(batch_size, 3), requires_grad=True)
        agent.policy.discrete_target_critic = _ConstQ(torch.full((batch_size, 3), 2.0))
        return agent

    rlpd_hybrid_module.F.mse_loss = spying_mse_loss
    try:
        _make(use_grasp_penalty=False)._train_discrete_critic(1)
        _make(use_grasp_penalty=True)._train_discrete_critic(1)
    finally:
        rlpd_hybrid_module.F.mse_loss = real_mse_loss

    target_unshaped, target_shaped = captured_targets
    torch.testing.assert_close(target_unshaped, torch.full((batch_size,), 3.0))  # 1.0 + 1*1.0*2.0
    torch.testing.assert_close(target_shaped, torch.full((batch_size,), 2.95))  # 0.95 + 2.0
    torch.testing.assert_close(target_shaped - target_unshaped, penalty)


_TEST_IMAGE_SIZE = 16


class ImageDummyVecEnv:
    """Dict obs space with an already-stacked (frame_stack, H, W, C) image
    key -- mimics what ImageFrameStackWrapper + FlattenRGBDObservationWrapper
    would produce upstream, for testing memory_efficient_buffer=True."""

    num_envs = 2

    def __init__(self, frame_stack: int = 3) -> None:
        self.frame_stack = frame_stack
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32),
                "rgb": spaces.Box(
                    0, 255, (frame_stack, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3), dtype=np.uint8
                ),
            }
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (self.num_envs, 3)),
            high=np.broadcast_to(self.single_action_space.high, (self.num_envs, 3)),
            dtype=np.float32,
        )

    def _obs(self):
        return {
            "state": torch.randn(self.num_envs, 4),
            "rgb": torch.randint(
                0, 256,
                (self.num_envs, self.frame_stack, _TEST_IMAGE_SIZE, _TEST_IMAGE_SIZE, 3),
                dtype=torch.uint8,
            ),
        }

    def reset(self, seed: int | None = None):
        del seed
        return self._obs(), {}

    def step(self, actions):
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def test_memory_efficient_buffer_constructs_the_dedup_buffer_and_learns():
    from rl_garden.buffers.memory_efficient_dict_buffer import MemoryEfficientDictReplayBuffer
    from rl_garden.encoders.combined import default_image_encoder_factory

    agent = RLPDHybrid(
        env=ImageDummyVecEnv(frame_stack=3),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
        image_keys=("rgb",),
        image_encoder_factory=default_image_encoder_factory(
            features_dim=16, plain_conv_pooling="gap"
        ),
        enable_stacking=True,
        memory_efficient_buffer=True,
        memory_efficient_image_keys=("rgb",),
        memory_efficient_frame_stack=3,
    )
    assert isinstance(agent.replay_buffer, MemoryEfficientDictReplayBuffer)
    agent.learn(total_timesteps=8)
    assert agent._global_step == 8


def test_checkpoint_round_trips_discrete_critic_and_dqn_optimizer():
    agent = _agent()
    agent.learn(total_timesteps=8)

    buf = io.BytesIO()
    torch.save(agent.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=False)

    assert "dqn_optimizer" in state["optimizers"]
    assert any("discrete_critic" in k for k in state["policy"].keys())

    fresh = _agent()
    fresh.policy.load_state_dict(state["policy"])
    fresh.dqn_optimizer.load_state_dict(state["optimizers"]["dqn_optimizer"])
