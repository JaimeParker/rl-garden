from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import PPO
from rl_garden.buffers import RolloutBuffer
from rl_garden.encoders import FlattenExtractor
from rl_garden.encoders.config import EncoderConfig
from rl_garden.policies.ppo_policy import PPOPolicy, get_ppo_arch


class DummyVecEnv:
    def __init__(
        self, observation_space: spaces.Space, action_space: spaces.Box
    ) -> None:
        self.num_envs = 2
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(
                action_space.low, (self.num_envs,) + action_space.shape
            ),
            high=np.broadcast_to(
                action_space.high, (self.num_envs,) + action_space.shape
            ),
            dtype=action_space.dtype,
        )
        self._step = 0

    def reset(self, seed: int | None = None):
        del seed
        self._step = 0
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)
        self._step += 1
        obs = self._obs()
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def _obs(self):
        if isinstance(self.single_observation_space, spaces.Dict):
            return {
                "rgb_cam": torch.randint(
                    0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8
                ),
                "state": torch.randn(self.num_envs, 4),
            }
        return torch.randn(self.num_envs, *self.single_observation_space.shape)


def _state_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


def _dict_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def _ppo_kwargs() -> dict[str, object]:
    return {
        "device": "cpu",
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "target_kl": None,
        "net_arch": [16],
    }


def test_get_ppo_arch_from_list_and_dict():
    assert get_ppo_arch([32, 16]) == ([32, 16], [32, 16])
    assert get_ppo_arch({"pi": [16], "vf": [32]}) == ([16], [32])


def test_ppo_policy_action_value_log_prob_shapes():
    obs_space = _state_space()
    action_space = _action_space()
    policy = PPOPolicy(
        obs_space,
        action_space,
        FlattenExtractor(obs_space),
        net_arch=[16],
    )
    obs = torch.randn(5, 4)
    actions, values, log_prob, entropy = policy(obs)

    assert actions.shape == (5, 2)
    assert values.shape == (5, 1)
    assert log_prob.shape == (5, 1)
    assert entropy.shape == (5, 1)

    values_eval, log_prob_eval, entropy_eval = policy.evaluate_actions(obs, actions)
    assert values_eval.shape == (5, 1)
    assert log_prob_eval.shape == (5, 1)
    assert entropy_eval.shape == (5, 1)


def test_ppo_policy_sum_dims_false_returns_per_dimension_shapes():
    obs_space = _state_space()
    action_space = _action_space()
    policy = PPOPolicy(
        obs_space, action_space, FlattenExtractor(obs_space), net_arch=[16]
    )
    obs = torch.randn(5, 4)
    actions, _values, summed_log_prob, summed_entropy = policy(obs, sum_dims=True)
    _actions2, _values2, unsummed_log_prob, unsummed_entropy = policy(
        obs, sum_dims=False
    )
    assert summed_log_prob.shape == (5, 1)
    assert unsummed_log_prob.shape == (5, 2)  # action_space has shape (2,)
    assert summed_entropy.shape == (5, 1)
    assert unsummed_entropy.shape == (5, 2)

    values_eval, unsummed_eval_log_prob, unsummed_eval_entropy = (
        policy.evaluate_actions(obs, actions, sum_dims=False)
    )
    assert values_eval.shape == (5, 1)
    assert unsummed_eval_log_prob.shape == (5, 2)
    assert unsummed_eval_entropy.shape == (5, 2)


def test_ppo_policy_act_with_value_and_logprob_matches_forward():
    obs_space = _state_space()
    action_space = _action_space()
    policy = PPOPolicy(
        obs_space, action_space, FlattenExtractor(obs_space), net_arch=[16]
    )
    obs = torch.randn(5, 4)
    actions, values, log_prob, entropy, state = policy.act_with_value_and_logprob(obs)
    assert actions.shape == (5, 2)
    assert values.shape == (5, 1)
    assert log_prob.shape == (5, 1)
    assert entropy.shape == (5, 1)
    assert state is None

    sentinel = object()
    *_rest, state_out = policy.act_with_value_and_logprob(obs, state=sentinel)
    assert state_out is sentinel  # no-op passthrough, not silently dropped


def test_rollout_buffer_computes_gae_returns():
    buffer = RolloutBuffer(
        _state_space(),
        _action_space(),
        num_steps=3,
        num_envs=2,
        device="cpu",
        gamma=1.0,
        gae_lambda=1.0,
    )
    for _ in range(3):
        buffer.add(
            torch.zeros(2, 4),
            torch.zeros(2, 2),
            torch.ones(2),
            torch.zeros(2),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        )

    buffer.compute_returns_and_advantage(torch.zeros(2), torch.zeros(2))

    assert torch.allclose(buffer.returns[0], torch.full((2,), 3.0))
    assert torch.allclose(buffer.returns[1], torch.full((2,), 2.0))
    assert torch.allclose(buffer.returns[2], torch.full((2,), 1.0))
    sample = next(buffer.get(batch_size=2))
    assert sample.obs.shape == (2, 4)
    assert sample.actions.shape == (2, 2)


def test_ppo_learn_one_iteration_state():
    env = DummyVecEnv(_state_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs())

    agent.learn(total_timesteps=4)

    assert agent._global_step == 4
    assert agent._global_update == 1


def test_ppo_dict_obs_constructs_and_trains_one_update():
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs())

    agent.learn(total_timesteps=4)

    # Default encoder_sharing="shared_critic_grad" with no separate
    # critic_extractor: BasePolicy.extract_actor_features's stop-gradient
    # rule applies (the old algorithm-level _actor_stop_gradient hook is
    # gone -- see rl_garden/policies/base.py).
    assert agent.policy.encoder_sharing == "shared_critic_grad"
    assert agent.policy.critic_extractor is None
    assert agent._global_step == 4


def test_ppo_checkpoint_roundtrip(tmp_path):
    env = DummyVecEnv(_state_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs())
    agent.learn(total_timesteps=4)
    path = tmp_path / "ppo.pt"
    agent.save(path)

    loaded = PPO(env=DummyVecEnv(_state_space(), _action_space()), **_ppo_kwargs())
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    assert loaded._global_update == agent._global_update
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


def test_ppo_dict_obs_checkpoint_roundtrip_with_encoder_config(tmp_path):
    encoder_config = EncoderConfig(proprio_latent_dim=4)
    env = DummyVecEnv(_dict_space(), _action_space())
    agent = PPO(env=env, **_ppo_kwargs(), encoder_config=encoder_config)
    agent.learn(total_timesteps=4)
    path = tmp_path / "ppo_dict.pt"
    agent.save(path)

    loaded = PPO(
        env=DummyVecEnv(_dict_space(), _action_space()),
        **_ppo_kwargs(),
        encoder_config=encoder_config,
    )
    loaded.load(path)

    assert loaded._global_step == agent._global_step
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


# --- policy-extractor-contract: asymmetric obs_groups end-to-end (state_<name>) ---


class _StateExtraVecEnv:
    """A ``rgb_cam`` + ``state`` + ``state_object_pose`` env: ``rgb_cam``/
    ``state`` are shared by actor and critic (both get a real, trainable
    ``CombinedExtractor``); ``state_object_pose`` (Section A's ``state_<name>``
    family) is critic-only."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "state_object_pose": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )
        act_space = _action_space()
        self.single_action_space = act_space
        self.action_space = spaces.Box(
            low=np.broadcast_to(act_space.low, (self.num_envs,) + act_space.shape),
            high=np.broadcast_to(act_space.high, (self.num_envs,) + act_space.shape),
            dtype=act_space.dtype,
        )

    def reset(self, seed: int | None = None):
        del seed
        return self._obs(), {}

    def step(self, actions):
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)
        obs = self._obs()
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None

    def _obs(self):
        return {
            "rgb_cam": torch.randint(0, 256, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(self.num_envs, 4),
            "state_object_pose": torch.randn(self.num_envs, 3),
        }


def test_ppo_asymmetric_obs_groups_critic_sees_extra_state_actor_does_not():
    """End-to-end asymmetric actor/critic encoders via state_<name>: critic
    sees state_object_pose, actor does not, encoder_sharing="separate".
    Asserts the actor extractor's schema lacks the key and, over one real
    rollout+update step, gradients reach only the right extractor."""
    from rl_garden.observations import ObsGroups

    agent = PPO(
        env=_StateExtraVecEnv(),
        device="cpu",
        num_steps=4,
        num_minibatches=1,
        update_epochs=1,
        eval_freq=0,
        log_freq=0,
        target_kl=None,
        net_arch=[16],
        encoder_config=EncoderConfig(proprio_latent_dim=4),
        obs_groups=ObsGroups(
            actor=("rgb_cam", "state"), critic=("rgb_cam", "state", "state_object_pose")
        ),
        encoder_sharing="separate",
    )

    actor_extractor = agent.policy.actor_extractor
    critic_extractor = agent.policy.critic_extractor
    assert critic_extractor is not None and critic_extractor is not actor_extractor
    assert "state_object_pose" not in actor_extractor.state_keys
    assert "state_object_pose" in critic_extractor.state_keys

    agent.learn(total_timesteps=4)

    data = next(agent._iter_minibatches())
    values, log_prob, _entropy = agent.policy.evaluate_actions(data.obs, data.actions)

    actor_grad_on_actor = torch.autograd.grad(
        log_prob.sum(), list(actor_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in actor_grad_on_actor)
    actor_grad_on_critic = torch.autograd.grad(
        log_prob.sum(), list(critic_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert all(g is None for g in actor_grad_on_critic)

    critic_grad_on_critic = torch.autograd.grad(
        values.sum(), list(critic_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert any(g is not None and torch.any(g != 0) for g in critic_grad_on_critic)
    critic_grad_on_actor = torch.autograd.grad(
        values.sum(), list(actor_extractor.parameters()), allow_unused=True
    )
    assert all(g is None for g in critic_grad_on_actor)
