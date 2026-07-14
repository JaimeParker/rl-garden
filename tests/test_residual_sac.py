from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import ResidualSAC
from rl_garden.buffers import ResidualDictReplayBuffer, ResidualTensorReplayBuffer
from rl_garden.common import ActionScaler
from rl_garden.common.types import ResidualReplayBufferSample
from rl_garden.policies.base_policies import BasePolicyOutput, BasePolicyProvider


class ConstantBaseProvider(BasePolicyProvider):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        action: torch.Tensor,
    ) -> None:
        super().__init__(observation_space, action_space, device="cpu")
        self.action = torch.nn.Parameter(action.float())
        self.reset_calls = 0
        self.select_calls = 0

    def select_action(self, obs):
        self.select_calls += 1
        if isinstance(obs, dict):
            n = next(iter(obs.values())).shape[0]
            device = next(iter(obs.values())).device
        else:
            n = obs.shape[0]
            device = obs.device
        return BasePolicyOutput(actions=self.action.to(device).expand(n, -1))

    def reset(self, env_ids=None) -> None:
        del env_ids
        self.reset_calls += 1


class RawActionVecEnv:
    def __init__(
        self,
        *,
        action_space: spaces.Box | None = None,
        num_envs: int = 2,
    ) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.single_action_space = action_space or spaces.Box(
            low=np.array([-0.1, -0.1], dtype=np.float32),
            high=np.array([0.1, 0.1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.broadcast_to(
                self.single_action_space.low,
                (self.num_envs,) + self.single_action_space.shape,
            ),
            high=np.broadcast_to(
                self.single_action_space.high,
                (self.num_envs,) + self.single_action_space.shape,
            ),
            dtype=np.float32,
        )
        self.last_actions = None

    def reset(self, seed: int | None = None):
        del seed
        return torch.zeros(self.num_envs, *self.single_observation_space.shape), {}

    def step(self, actions):
        self.last_actions = actions.detach().clone()
        obs = torch.ones(self.num_envs, *self.single_observation_space.shape)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}


def _agent(env=None, base_action=None, **kwargs) -> ResidualSAC:
    env = env or RawActionVecEnv()
    provider = ConstantBaseProvider(
        env.single_observation_space,
        env.single_action_space,
        torch.tensor([0.05, -0.05]) if base_action is None else base_action,
    )
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 16,
        "batch_size": 4,
        "learning_starts": 100,
        "training_freq": 2,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [16],
        "residual_action_scale": 0.0,
    }
    params.update(kwargs)
    return ResidualSAC(env=env, base_action_provider=provider, **params)


def _raw_joint_base_action() -> torch.Tensor:
    return torch.tensor(
        [
            1.2,
            -1.4,
            0.3,
            -0.4,
            0.5,
            -0.6,
            0.75,
            -1.1,
            1.3,
            -0.2,
            0.4,
            -0.5,
            0.6,
            0.25,
        ]
    )


def _raw_joint_agent(base_action=None, **kwargs) -> ResidualSAC:
    action_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(14,),
        dtype=np.float32,
    )
    env = RawActionVecEnv(action_space=action_space, num_envs=1)
    params = {
        "residual_action_coordinates": "raw_joint_delta",
        "residual_action_scale": 0.1,
        "joint_delta_scale": 0.05,
        "gripper_delta_scale": 0.2,
    }
    params.update(kwargs)
    return _agent(
        env=env,
        base_action=_raw_joint_base_action() if base_action is None else base_action,
        **params,
    )


def _raw_training_batch(agent: ResidualSAC) -> ResidualReplayBufferSample:
    base = _raw_joint_base_action().unsqueeze(0)
    next_base = base.clone()
    next_base[..., :6] += 0.02
    final_action = base.clone()
    final_action[..., 7:13] -= 0.01
    for _ in range(agent.batch_size):
        agent.replay_buffer.add(
            torch.zeros(agent.num_envs, 3),
            torch.ones(agent.num_envs, 3),
            final_action,
            torch.ones(agent.num_envs),
            torch.zeros(agent.num_envs),
            base_actions=base,
            next_base_actions=next_base,
        )
    return agent.replay_buffer.sample(agent.batch_size)


def _expected_raw_composition(
    base_actions: torch.Tensor,
    unit_residual: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    scale = torch.tensor(
        [0.05] * 6 + [0.2] + [0.05] * 6 + [0.2],
        dtype=base_actions.dtype,
        device=base_actions.device,
    ) * global_scale
    expected = base_actions + unit_residual * scale
    expected[..., 6] = expected[..., 6].clamp(0.0, 1.0)
    expected[..., 13] = expected[..., 13].clamp(0.0, 1.0)
    return expected


def _patch_unit_residual_actor(monkeypatch, agent: ResidualSAC, value: float):
    def actor_action_log_prob(
        _obs,
        base_actions,
        stop_gradient=False,
        detach_encoder=None,
    ):
        del _obs, stop_gradient, detach_encoder
        unit_residual = torch.full_like(base_actions, value)
        log_prob = torch.zeros(
            (base_actions.shape[0], 1),
            dtype=base_actions.dtype,
            device=base_actions.device,
        )
        features = torch.zeros(
            (base_actions.shape[0], 1),
            dtype=base_actions.dtype,
            device=base_actions.device,
        )
        return unit_residual, log_prob, features

    monkeypatch.setattr(
        agent.policy,
        "actor_action_log_prob",
        actor_action_log_prob,
    )


def test_action_scaler_scales_and_unscales_raw_actions():
    action_space = spaces.Box(
        low=np.array([-0.1, -0.2], dtype=np.float32),
        high=np.array([0.1, 0.2], dtype=np.float32),
        dtype=np.float32,
    )
    scaler = ActionScaler.from_action_space(action_space)

    raw = torch.tensor([[-0.1, 0.0], [0.0, 0.2]])
    normalized = scaler.scale(raw)
    assert torch.allclose(normalized, torch.tensor([[-1.0, 0.0], [0.0, 1.0]]))
    assert torch.allclose(scaler.unscale(normalized), raw)


def test_raw_joint_delta_builds_exact_finite_scale_vector():
    agent = _raw_joint_agent(residual_action_scale=0.1)

    expected = torch.tensor(
        [0.05] * 6 + [0.2] + [0.05] * 6 + [0.2]
    ) * 0.1

    assert agent.action_scaler is None
    torch.testing.assert_close(agent.resolved_residual_scale, expected)


def test_raw_joint_action_sanitizes_only_grippers():
    agent = _raw_joint_agent()
    action = _raw_joint_base_action()
    action[6] = 1.011
    action[13] = -0.01

    sanitized = agent._sanitize_raw_joint_action(action)

    torch.testing.assert_close(sanitized[:6], action[:6])
    torch.testing.assert_close(sanitized[7:13], action[7:13])
    assert sanitized[6].item() == 1.0
    assert sanitized[13].item() == 0.0
    assert sanitized.data_ptr() != action.data_ptr()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_raw_joint_action_rejects_non_finite_values(value: float):
    agent = _raw_joint_agent()
    action = _raw_joint_base_action()
    action[0] = value

    with pytest.raises(ValueError, match="non-finite"):
        agent._sanitize_raw_joint_action(action)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("residual_action_scale", float("nan"), "residual_action_scale"),
        ("residual_action_scale", float("inf"), "residual_action_scale"),
        ("residual_action_scale", -0.1, "residual_action_scale"),
        ("joint_delta_scale", float("nan"), "delta scales"),
        ("joint_delta_scale", -0.05, "delta scales"),
        ("gripper_delta_scale", float("inf"), "delta scales"),
        ("gripper_delta_scale", -0.2, "delta scales"),
    ],
)
def test_raw_scales_must_be_finite_and_non_negative(
    field: str, value: float, message: str
):
    with pytest.raises(ValueError, match=message):
        _raw_joint_agent(**{field: value})


def test_raw_joint_delta_requires_14_action_dimensions():
    with pytest.raises(ValueError, match="14-dimensional"):
        _agent(residual_action_coordinates="raw_joint_delta")


def test_raw_rollout_composes_exact_nonzero_correction(monkeypatch):
    agent = _raw_joint_agent(residual_action_scale=0.1)
    obs, _ = agent.env.reset()
    unit_residual = torch.tensor(
        [[1.0, -1.0] * 7],
        dtype=torch.float32,
    )

    def predict(_obs, *, base_actions, deterministic):
        del _obs, base_actions, deterministic
        return unit_residual

    monkeypatch.setattr(agent.policy, "predict", predict)

    replay_action, env_action, context = agent._rollout_action(
        obs, learning_has_started=True
    )

    base = _raw_joint_base_action().unsqueeze(0)
    expected = _expected_raw_composition(
        base,
        unit_residual,
        global_scale=0.1,
    )
    torch.testing.assert_close(replay_action, expected)
    torch.testing.assert_close(env_action, expected)
    assert context is not None
    torch.testing.assert_close(context["base_actions"], base)


def test_raw_rollout_stores_executed_action_and_sanitized_base_in_replay():
    base_action = _raw_joint_base_action()
    base_action[6] = 1.011
    base_action[13] = -0.01
    agent = _raw_joint_agent(
        base_action=base_action,
        residual_action_scale=0.0,
        training_freq=1,
    )

    agent.learn(total_timesteps=1)

    expected = base_action.unsqueeze(0).clone()
    expected[..., 6] = 1.0
    expected[..., 13] = 0.0
    torch.testing.assert_close(agent.env.last_actions, expected)
    torch.testing.assert_close(agent.replay_buffer.actions[0], expected)
    torch.testing.assert_close(agent.replay_buffer.base_actions[0], expected)
    torch.testing.assert_close(agent.replay_buffer.next_base_actions[0], expected)


def test_raw_rollout_reuses_cached_next_base_action():
    agent = _raw_joint_agent(residual_action_scale=0.0)
    obs, _ = agent.env.reset()
    agent._on_env_reset(obs)

    _, env_action, context = agent._rollout_action(
        obs, learning_has_started=False
    )
    next_obs, _, _, _, infos = agent.env.step(env_action)
    replay_kwargs = agent._replay_buffer_add_kwargs(
        context,
        obs,
        next_obs,
        next_obs,
        infos,
        torch.zeros(agent.num_envs, dtype=torch.bool),
    )
    calls_after_next_base = agent.base_action_provider.select_calls

    _, _, next_context = agent._rollout_action(
        next_obs, learning_has_started=False
    )

    assert agent.base_action_provider.select_calls == calls_after_next_base
    assert next_context is not None
    torch.testing.assert_close(
        next_context["base_actions"], replay_kwargs["next_base_actions"]
    )


def test_raw_critic_uses_stored_final_raw_replay_action(monkeypatch):
    agent = _raw_joint_agent(learning_starts=0)
    data = _raw_training_batch(agent)
    captured = {}

    def critic_forward(_obs, actions, target=False):
        del _obs
        captured["actions"] = actions.detach().clone()
        captured["target"] = target
        q_value = actions.sum(dim=-1, keepdim=True)
        return q_value.unsqueeze(0).expand(agent.n_critics, -1, -1)

    def target_q(batch):
        return torch.zeros(
            (batch.actions.shape[0], 1),
            dtype=batch.actions.dtype,
            device=batch.actions.device,
        )

    monkeypatch.setattr(agent, "_critic_forward", critic_forward)
    monkeypatch.setattr(agent, "_target_q", target_q)

    critic_loss, _ = agent._critic_loss(data)

    assert critic_loss.shape == ()
    assert captured["target"] is False
    assert captured["actions"].shape[-1] == 14
    torch.testing.assert_close(captured["actions"], data.actions)


def test_raw_actor_loss_critic_action_composes_current_base(monkeypatch):
    agent = _raw_joint_agent(learning_starts=0, residual_action_scale=0.1)
    data = _raw_training_batch(agent)
    _patch_unit_residual_actor(monkeypatch, agent, value=0.5)
    captured = {}

    def min_q_value(features, actions, subsample_size, target):
        del features, subsample_size
        captured["actions"] = actions.detach().clone()
        captured["target"] = target
        return actions.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(agent.policy, "min_q_value", min_q_value)

    actor_loss, _ = agent._actor_loss_from_batch(data)

    unit_residual = torch.full_like(data.base_actions, 0.5)
    expected = _expected_raw_composition(
        data.base_actions,
        unit_residual,
        global_scale=0.1,
    )
    assert actor_loss.shape == ()
    assert captured["target"] is False
    assert captured["actions"].shape[-1] == 14
    torch.testing.assert_close(captured["actions"], expected)


def test_raw_target_critic_action_composes_next_base(monkeypatch):
    agent = _raw_joint_agent(learning_starts=0, residual_action_scale=0.1)
    data = _raw_training_batch(agent)
    _patch_unit_residual_actor(monkeypatch, agent, value=-0.25)
    captured = {}

    def min_q_value(features, actions, subsample_size, target):
        del features, subsample_size
        captured["actions"] = actions.detach().clone()
        captured["target"] = target
        return actions.sum(dim=-1, keepdim=True)

    monkeypatch.setattr(agent.policy, "min_q_value", min_q_value)

    target_q = agent._target_q(data)

    unit_residual = torch.full_like(data.next_base_actions, -0.25)
    expected = _expected_raw_composition(
        data.next_base_actions,
        unit_residual,
        global_scale=0.1,
    )
    assert target_q.shape == (agent.batch_size, 1)
    assert captured["target"] is True
    assert captured["actions"].shape[-1] == 14
    torch.testing.assert_close(captured["actions"], expected)


def test_raw_actor_backward_does_not_train_base_policy():
    agent = _raw_joint_agent(learning_starts=0, residual_action_scale=0.1)
    data = _raw_training_batch(agent)
    provider = agent.base_action_provider
    provider_calls = provider.select_calls
    agent.actor_optimizer.zero_grad(set_to_none=True)

    actor_loss, _ = agent._actor_loss_from_batch(data)
    actor_loss.backward()

    actor_grads = [param.grad for param in agent.policy.actor_parameters()]
    provider_param_ids = {id(param) for param in provider.parameters()}
    optimizer_param_ids = {
        id(param)
        for optimizer in (agent.actor_optimizer, agent.q_optimizer)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert any(grad is not None for grad in actor_grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in actor_grads)
    assert provider.action.grad is None
    assert provider_param_ids.isdisjoint(optimizer_param_ids)
    assert provider.select_calls == provider_calls


def test_residual_tensor_replay_buffer_samples_base_actions():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = ResidualTensorReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")

    rb.add(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.zeros(2, 2),
        torch.ones(2),
        torch.zeros(2),
        base_actions=torch.full((2, 2), 0.25),
        next_base_actions=torch.full((2, 2), -0.25),
    )

    batch = rb.sample(2)
    assert batch.base_actions.shape == (2, 2)
    assert batch.next_base_actions.shape == (2, 2)
    assert torch.all(batch.base_actions == 0.25)
    assert torch.all(batch.next_base_actions == -0.25)


def test_residual_dict_replay_buffer_samples_base_actions():
    obs_space = spaces.Dict(
        {"state": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)}
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    rb = ResidualDictReplayBuffer(obs_space, act_space, 2, 8, "cpu", "cpu")
    obs = {"state": torch.zeros(2, 3)}
    next_obs = {"state": torch.ones(2, 3)}

    rb.add(
        obs,
        next_obs,
        torch.zeros(2, 2),
        torch.ones(2),
        torch.zeros(2),
        base_actions=torch.full((2, 2), 0.5),
        next_base_actions=torch.full((2, 2), -0.5),
    )

    batch = rb.sample(2)
    assert batch.obs["state"].shape == (2, 3)
    assert batch.base_actions.shape == (2, 2)
    assert batch.next_base_actions.shape == (2, 2)


def test_residual_rollout_unscales_env_action_and_stores_normalized_action():
    env = RawActionVecEnv()
    agent = _agent(env=env, residual_action_scale=0.0)

    agent.learn(total_timesteps=2)

    expected_raw = torch.tensor([[0.05, -0.05], [0.05, -0.05]])
    expected_normalized = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    assert torch.allclose(env.last_actions, expected_raw)
    assert torch.allclose(agent.replay_buffer.actions[0], expected_normalized)
    assert torch.allclose(agent.replay_buffer.base_actions[0], expected_normalized)
    assert torch.allclose(agent.replay_buffer.next_base_actions[0], expected_normalized)


def test_residual_update_hooks_combine_base_and_residual_actions():
    agent = _agent(learning_starts=0)
    env = agent.env
    for _ in range(4):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        base = torch.full((env.num_envs, 2), 0.25)
        next_base = torch.full((env.num_envs, 2), -0.25)
        agent.replay_buffer.add(
            obs,
            next_obs,
            base,
            torch.ones(env.num_envs),
            torch.zeros(env.num_envs),
            base_actions=base,
            next_base_actions=next_base,
        )

    data = agent.replay_buffer.sample(agent.batch_size)
    target_action, _, _ = agent._target_action_log_prob(data)
    actor_loss, log_prob = agent._actor_loss_from_batch(data)
    train_info = agent.train(1, compute_info=True)

    assert torch.allclose(target_action, data.next_base_actions)
    assert actor_loss.shape == ()
    assert log_prob.shape == (agent.batch_size, 1)
    assert torch.isfinite(torch.tensor(train_info["critic_loss"]))


def test_residual_critic_configuration_is_forwarded_to_policy():
    agent = _agent(n_critics=4, critic_subsample_size=2, critic_impl="legacy")

    assert agent.n_critics == 4
    assert agent.critic_subsample_size == 2
    assert agent.critic_impl == "legacy"
    assert agent.policy.n_critics == 4
    assert agent.policy.critic_subsample_size == 2
    assert agent.policy.critic.critic_impl == "legacy"


def test_residual_eval_q_mc_uses_normalized_final_action_for_critic():
    agent = _agent(residual_action_scale=0.0)
    obs, _ = agent.env.reset()

    env_action, critic_action = agent._eval_action_and_critic_action(obs)

    expected_env_action = torch.tensor([[0.05, -0.05], [0.05, -0.05]])
    expected_critic_action = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    torch.testing.assert_close(env_action, expected_env_action)
    torch.testing.assert_close(critic_action, expected_critic_action)


def test_residual_actor_diagnostics_use_base_actions_without_advancing_rng():
    agent = _agent()
    env = agent.env
    for _ in range(4):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        base = torch.full((env.num_envs, 2), 0.25)
        next_base = torch.full((env.num_envs, 2), -0.25)
        agent.replay_buffer.add(
            obs,
            next_obs,
            base,
            torch.ones(env.num_envs),
            torch.zeros(env.num_envs),
            base_actions=base,
            next_base_actions=next_base,
        )

    data = agent.replay_buffer.sample(agent.batch_size)

    torch.manual_seed(123)
    expected = torch.rand(4)

    torch.manual_seed(123)
    diagnostics = agent._actor_diagnostics(data)
    actual = torch.rand(4)

    assert "action_saturation" in diagnostics
    assert "entropy_gaussian" in diagnostics
    torch.testing.assert_close(actual, expected)


def test_residual_sample_train_batch_mixes_online_and_offline_samples():
    agent = _agent()
    agent.offline_replay_buffer = agent._make_residual_replay_buffer(16)
    agent.offline_data_ratio = 0.5
    env = agent.env
    for replay_buffer in (agent.replay_buffer, agent.offline_replay_buffer):
        replay_buffer.add(
            torch.zeros(env.num_envs, 3),
            torch.ones(env.num_envs, 3),
            torch.zeros(env.num_envs, 2),
            torch.zeros(env.num_envs),
            torch.zeros(env.num_envs),
            base_actions=torch.zeros(env.num_envs, 2),
            next_base_actions=torch.zeros(env.num_envs, 2),
        )

    calls: list[tuple[str, int]] = []

    def sample_from(name: str, value: float):
        def _sample(batch_size: int) -> ResidualReplayBufferSample:
            calls.append((name, batch_size))
            return ResidualReplayBufferSample(
                obs=torch.full((batch_size, 3), value),
                next_obs=torch.full((batch_size, 3), value + 0.1),
                actions=torch.full((batch_size, 2), value),
                rewards=torch.full((batch_size,), value),
                dones=torch.zeros(batch_size),
                base_actions=torch.full((batch_size, 2), value + 0.2),
                next_base_actions=torch.full((batch_size, 2), value + 0.3),
            )

        return _sample

    agent.replay_buffer.sample = sample_from("online", 1.0)
    agent.offline_replay_buffer.sample = sample_from("offline", 2.0)

    batch = agent._sample_train_batch(4)

    assert calls == [("online", 2), ("offline", 2)]
    torch.testing.assert_close(batch.obs[:2], torch.full((2, 3), 1.0))
    torch.testing.assert_close(batch.obs[2:], torch.full((2, 3), 2.0))
    torch.testing.assert_close(batch.base_actions[:2], torch.full((2, 2), 1.2))
    torch.testing.assert_close(batch.base_actions[2:], torch.full((2, 2), 2.2))


def test_residual_offline_buffer_defaults_to_loadable_dataset_size(tmp_path):
    path = tmp_path / "residual_demo.h5"
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = "rl_garden_residual_offline"
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=np.ones((5, 3), dtype=np.float32))
        group.create_dataset("actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset("base_actions", data=np.ones((4, 2), dtype=np.float32))
        group.create_dataset(
            "next_base_actions", data=np.ones((4, 2), dtype=np.float32)
        )
        group.create_dataset("rewards", data=np.ones(4, dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, False, True]))
        group.create_dataset("truncated", data=np.array([False, False, False, False]))

    agent = _agent(buffer_size=128)

    loaded = agent.load_offline_replay_buffer(path)

    assert loaded == 4
    assert agent.offline_replay_buffer.num_envs == 1
    assert agent.offline_replay_buffer.buffer_size == 4
    assert agent.offline_replay_buffer.per_env_buffer_size == 4
    assert len(agent.offline_replay_buffer) == 4
