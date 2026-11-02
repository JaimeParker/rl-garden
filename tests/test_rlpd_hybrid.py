from __future__ import annotations

import io
import types

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
from rl_garden.buffers.replay_buffer import ReplayBuffer


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
    kwargs = {
        "env": DummyVecEnv(),
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 1,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [8],
        "discrete_hidden_dim": 8,
    }
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
    action = agent.policy.predict({"state": torch.zeros(2, 4)}, deterministic=True)
    assert action.shape == (2, 3)


def test_learns_with_demo_buffer_mixed_in():
    agent = _agent()
    agent.init_demo_buffer(buffer_size=32, demo_data_ratio=0.5)

    obs = {"state": torch.zeros(4)}
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


def test_use_grasp_penalty_with_box_env_no_longer_raises():
    # DummyVecEnv's bare Box observation space is boundary-normalized to
    # Dict by BaseAlgorithm.__init__ (see rl_garden.envs.wrappers
    # .VectorizedDictStateWrapper) before RLPDHybrid ever sees it, so
    # use_grasp_penalty=True no longer requires an explicitly Dict env.
    agent = _agent(use_grasp_penalty=True)
    assert isinstance(agent.replay_buffer, ReplayBuffer)


def test_use_grasp_penalty_with_nstep_greater_than_one_raises():
    with pytest.raises(ValueError, match="nstep == 1"):
        _agent(use_grasp_penalty=True, nstep=2)


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
                "rgb_cam": spaces.Box(
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
            "rgb_cam": torch.randint(
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
    from rl_garden.buffers.memory_efficient_buffer import MemoryEfficientReplayBuffer
    from rl_garden.encoders.config import EncoderConfig

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
        encoder_config=EncoderConfig(features_dim=16, plain_conv_pooling="gap"),
        memory_efficient_buffer=True,
        memory_efficient_frame_stack=3,
    )
    assert isinstance(agent.replay_buffer, MemoryEfficientReplayBuffer)
    agent.learn(total_timesteps=8)
    assert agent._global_step == 8


class _TruncatingImageDummyVecEnv(ImageDummyVecEnv):
    """Truncates (never terminates) env 0 every other step."""

    def __init__(self, frame_stack: int = 3) -> None:
        super().__init__(frame_stack=frame_stack)
        self._steps = 0

    def step(self, actions):
        next_obs, rewards, terminations, truncations, infos = super().step(actions)
        self._steps += 1
        truncations[0] = self._steps % 2 == 0
        return next_obs, rewards, terminations, truncations, infos


def test_memory_efficient_buffer_receives_truncations_as_episode_end():
    from rl_garden.encoders.config import EncoderConfig

    agent = RLPDHybrid(
        env=_TruncatingImageDummyVecEnv(frame_stack=3),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=100,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
        encoder_config=EncoderConfig(features_dim=16, plain_conv_pooling="gap"),
        memory_efficient_buffer=True,
        memory_efficient_frame_stack=3,
    )
    recorded = []
    original_add = agent.replay_buffer.add

    def _recording_add(obs, next_obs, action, reward, done, **kwargs):
        recorded.append((done.clone(), kwargs.get("episode_end")))
        original_add(obs, next_obs, action, reward, done, **kwargs)

    agent.replay_buffer.add = _recording_add
    agent.learn(total_timesteps=8)

    assert len(recorded) == 4
    for step, (done, episode_end) in enumerate(recorded, start=1):
        expected = torch.tensor([step % 2 == 0, False])
        assert torch.equal(episode_end, expected)
        # Under bootstrap_at_done="truncated" the stored done stays False.
        assert not bool(done.bool().any())
    # Env 0 truncated twice, so the buffer opened two new episodes for it.
    assert agent.replay_buffer._current_ep_id.tolist() == [2, 0]


def test_replay_buffer_step_kwargs_omit_episode_end_without_memory_efficient_buffer():
    agent = _agent()
    terminations = torch.tensor([True, False])
    truncations = torch.tensor([False, True])
    assert agent._replay_buffer_step_kwargs(terminations, truncations) == {}


def test_checkpoint_round_trips_discrete_critic_and_dqn_optimizer():
    agent = _agent()
    agent.learn(total_timesteps=8)

    buf = io.BytesIO()
    torch.save(agent.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=False)

    assert "dqn_optimizer" in state["optimizers"]
    assert any("discrete_critic" in k for k in state["policy"])

    fresh = _agent()
    fresh.policy.load_state_dict(state["policy"])
    fresh.dqn_optimizer.load_state_dict(state["optimizers"]["dqn_optimizer"])


# --- num_grippers: the last k action dims are grippers ---------------------------------


def test_one_gripper_labels_keep_the_nearest_of_three_mapping():
    from rl_garden.algorithms.rlpd_hybrid import _discrete_labels_from_actions
    from rl_garden.policies.rlpd_hybrid_policy import encode_gripper_labels

    gripper = torch.tensor([-1.0, -0.6, -0.5, -0.2, 0.0, 0.3, 0.5, 0.7, 1.0, 2.0, -3.0])
    actions = torch.zeros(gripper.shape[0], 3)
    actions[:, -1] = gripper
    expected = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0]  # ties go to the lower value

    nearest_of_three = (actions[..., -1:] - torch.tensor([-1.0, 0.0, 1.0])).abs().argmin(dim=-1)
    assert nearest_of_three.tolist() == expected
    labels = _discrete_labels_from_actions(actions)
    assert labels.tolist() == expected and labels.dtype == torch.int64
    assert encode_gripper_labels(actions[..., -1:]).tolist() == expected


def test_two_gripper_labels_round_trip_all_nine_joint_actions():
    from rl_garden.policies.rlpd_hybrid_policy import (
        decode_gripper_index,
        encode_gripper_labels,
    )

    index = torch.arange(9)
    values = decode_gripper_index(index, 2)
    assert values.shape == (9, 2)
    expected = [(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]  # first gripper most significant
    assert [tuple(row) for row in values.tolist()] == expected
    assert encode_gripper_labels(values).tolist() == index.tolist()
    # Non-exact gripper actions map to the nearest values per gripper.
    assert encode_gripper_labels(torch.tensor([[0.9, -0.2], [-0.7, 0.6]])).tolist() == [7, 2]


def test_one_gripper_decode_is_the_value_table():
    from rl_garden.policies.rlpd_hybrid_policy import decode_gripper_index

    decoded = decode_gripper_index(torch.tensor([0, 1, 2]), 1)
    assert decoded.shape == (3, 1)
    assert decoded.reshape(-1).tolist() == [-1.0, 0.0, 1.0]


class TwoGripperVecEnv(DummyVecEnv):
    """6D action space: two 2-D arm blocks, then the left and right grippers."""

    ACTION_DIM = 6

    def __init__(self, num_envs: int = 2) -> None:
        super().__init__(num_envs)
        dim = self.ACTION_DIM
        self.single_action_space = spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, dim)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, dim)),
            dtype=np.float32,
        )


def test_one_gripper_head_width_policy_kwargs_and_metadata_are_unchanged():
    agent = _agent()
    assert agent.num_grippers == 1 and agent.policy.num_grippers == 1
    assert agent.policy.discrete_critic(torch.zeros(1, agent.policy.critic_features_dim)).shape == (1, 3)
    assert agent.policy.action_space.shape == (2,)
    assert "num_grippers" not in agent._policy_kwargs(agent.env.single_observation_space)
    assert agent._checkpoint_metadata()["num_grippers"] == 1


def test_two_grippers_set_head_width_continuous_dims_and_metadata():
    agent = _agent(env=TwoGripperVecEnv(), num_grippers=2)
    policy = agent.policy
    assert policy.num_grippers == 2
    assert policy.action_space.shape == (4,)
    assert policy.discrete_critic(torch.zeros(1, policy.critic_features_dim)).shape == (1, 9)
    assert policy.discrete_target_critic(torch.zeros(1, policy.critic_features_dim)).shape == (1, 9)
    assert agent._checkpoint_metadata()["num_grippers"] == 2


def test_two_gripper_predict_appends_the_decoded_grippers_after_the_continuous_dims():
    agent = _agent(env=TwoGripperVecEnv(), num_grippers=2)
    q_values = torch.zeros(2, 9)
    q_values[:, 5] = 1.0  # digits (1, 2): left hold, right open
    agent.policy.discrete_critic = _ConstQ(q_values)

    action = agent.policy.predict({"state": torch.zeros(2, 4)}, deterministic=True)

    assert action.shape == (2, TwoGripperVecEnv.ACTION_DIM)
    torch.testing.assert_close(action[:, -2:], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    assert bool((action[:, :4].abs() <= 1.0).all())


def test_two_gripper_discrete_critic_is_trained_on_the_joint_label_of_the_last_two_dims():
    import rl_garden.algorithms.rlpd_hybrid as rlpd_hybrid_module

    actions = torch.zeros(4, TwoGripperVecEnv.ACTION_DIM)
    actions[:, -2:] = torch.tensor([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    batch = types.SimpleNamespace(
        obs={"state": torch.zeros(4, 4)},
        next_obs={"state": torch.zeros(4, 4)},
        actions=actions,
        rewards=torch.zeros(4),
        dones=torch.zeros(4),
    )
    agent = _agent(env=TwoGripperVecEnv(), num_grippers=2)
    agent._sample_train_batch = lambda n: batch
    # Q(s, j) = j, so the gathered prediction is the label itself.
    agent.policy.discrete_critic = _ConstQ(torch.arange(9.0).repeat(4, 1), requires_grad=True)
    agent.policy.discrete_target_critic = _ConstQ(torch.zeros(4, 9))

    captured = []
    real_mse_loss = rlpd_hybrid_module.F.mse_loss

    def spying_mse_loss(q_pred, target):
        captured.append(q_pred.detach().clone())
        return real_mse_loss(q_pred, target)

    rlpd_hybrid_module.F.mse_loss = spying_mse_loss
    try:
        agent._train_discrete_critic(1)
    finally:
        rlpd_hybrid_module.F.mse_loss = real_mse_loss

    assert captured[0].tolist() == [0.0, 7.0, 5.0, 8.0]


def test_two_gripper_critic_scores_the_continuous_dims_and_the_agent_learns():
    agent = _agent(env=TwoGripperVecEnv(), num_grippers=2)
    obs = {"state": torch.zeros(3, 4)}
    actions = torch.rand(3, TwoGripperVecEnv.ACTION_DIM) * 2 - 1
    with torch.no_grad():
        q_full = agent._critic_forward(obs, actions)
        q_continuous = super(RLPDHybrid, agent)._critic_forward(obs, actions[..., :4])
    torch.testing.assert_close(q_full, q_continuous)

    agent.learn(total_timesteps=16)
    assert agent._global_step == 16


@pytest.mark.parametrize("num_grippers", [0, -1])
def test_num_grippers_below_one_is_rejected(num_grippers):
    with pytest.raises(ValueError, match="num_grippers"):
        _agent(num_grippers=num_grippers)


def test_n_discrete_actions_must_match_the_gripper_count():
    from rl_garden.policies.rlpd_hybrid_policy import RLPDHybridPolicy

    agent = _agent(env=TwoGripperVecEnv(), num_grippers=2)
    kwargs = agent._policy_kwargs(agent.env.single_observation_space)
    with pytest.raises(ValueError, match="n_discrete_actions must be 9"):
        RLPDHybridPolicy(**{**kwargs, "n_discrete_actions": 3})
