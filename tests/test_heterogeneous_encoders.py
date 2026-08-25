"""Opt-in heterogeneous actor/critic encoders: default path stays shared and
byte-identical; a separate critic_features_extractor is correctly isolated
to the right optimizer/gradient path in both SAC-family and PPO-family."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import SAC
from rl_garden.common.training_phase import InitialTrainingPhase
from rl_garden.encoders import BaseFeaturesExtractor, CombinedExtractor
from rl_garden.networks.recurrent import RecurrentLatentEncoder
from rl_garden.policies.ppo_policy import PPOPolicy
from rl_garden.policies.recurrent_ppo_policy import RecurrentPPOPolicy
from rl_garden.policies.recurrent_sac_policy import RecurrentSACPolicy
from rl_garden.policies.rlpd_hybrid_policy import RLPDHybridPolicy
from rl_garden.policies.sac_policy import SACPolicy

class DummyVecEnv:
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Box) -> None:
        self.num_envs = 1
        self.single_observation_space = observation_space
        self.single_action_space = action_space
        self.action_space = action_space


class RecordingExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Space,
        features_dim: int = 13,
        marker: str = "default",
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker

    def forward(self, obs):
        batch = obs.shape[0] if isinstance(obs, torch.Tensor) else next(iter(obs.values())).shape[0]
        return torch.zeros(batch, self.features_dim)


class _TrainableDictExtractor(BaseFeaturesExtractor):
    """Matches _dict_env()'s rgb(8,8,3)/state(4,) shapes with real params,
    unlike RecordingExtractor (constant zeros, no gradient path)."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 6, marker: str = "") -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.marker = marker
        self.state_proj = torch.nn.Linear(4, features_dim)
        self.rgb_proj = torch.nn.Linear(3, features_dim)

    def forward(self, obs):
        state = obs["state"].float()
        rgb = obs["rgb"].float().mean(dim=(1, 2)) / 255.0
        return torch.tanh(self.state_proj(state) + self.rgb_proj(rgb))


class _TrainableBoxExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 6) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.proj = torch.nn.Linear(int(np.prod(observation_space.shape)), features_dim)

    def forward(self, obs):
        return torch.tanh(self.proj(obs.float()))


def _dict_env() -> DummyVecEnv:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DummyVecEnv(obs_space, act_space)


def _fill_dict(agent, steps: int = 8) -> None:
    env = agent.env
    for _ in range(steps):
        obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (env.num_envs, 64, 64, 3), dtype=torch.uint8),
            "state": torch.randn(env.num_envs, 4),
        }
        actions = torch.randn(env.num_envs, *env.single_action_space.shape).clamp(-1, 1)
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _clone_params(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _params_changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    assert list(module.parameters()), "module has no parameters -- test would trivially pass"
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


def _dict_agent(**kwargs) -> SAC:
    params = {
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 4,
        "learning_starts": 0,
        "training_freq": 1,
        "eval_freq": 0,
        "net_arch": {"pi": [16], "qf": [16]},
        "image_keys": ("rgb",),
        "proprio_latent_dim": 4,
    }
    params.update(kwargs)
    return SAC(env=_dict_env(), **params)


def _separate_critic_policy_kwargs(dim: int = 9) -> dict:
    return {
        "critic_features_extractor_class": _TrainableDictExtractor,
        "critic_features_extractor_kwargs": {"features_dim": dim, "marker": "critic"},
    }


# --- default path: identity, not just equal config ---


def test_default_path_critic_extractor_is_same_object_as_actor():
    agent = _dict_agent()
    assert agent.policy.critic_features_extractor is agent.policy.features_extractor


def test_default_path_actor_parameters_excludes_shared_encoder():
    agent = _dict_agent()
    actor_param_ids = {id(p) for p in agent.policy.actor_parameters()}
    encoder_param_ids = {id(p) for p in agent.policy.features_extractor.parameters()}
    assert not (actor_param_ids & encoder_param_ids)


# --- separate critic extractor: SAC gating ---


def test_separate_critic_extractor_updates_only_under_critic_loss():
    agent = _dict_agent(
        policy_kwargs=_separate_critic_policy_kwargs(),
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000, update_actor=False, update_critic=True, update_encoder=True
        ),
    )
    agent._start_initial_training_phase()
    _fill_dict(agent)

    assert isinstance(agent.policy.critic_features_extractor, _TrainableDictExtractor)
    assert agent.policy.critic_features_extractor is not agent.policy.features_extractor

    actor_encoder_before = _clone_params(agent.policy.features_extractor)
    critic_encoder_before = _clone_params(agent.policy.critic_features_extractor)

    agent.train(gradient_steps=1, compute_info=True)

    assert not _params_changed(actor_encoder_before, agent.policy.features_extractor)
    assert _params_changed(critic_encoder_before, agent.policy.critic_features_extractor)


def test_separate_actor_encoder_trains_via_actor_loss_when_not_shared():
    agent = _dict_agent(
        policy_kwargs=_separate_critic_policy_kwargs(),
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000, update_actor=True, update_critic=False, update_encoder=False
        ),
    )
    agent._start_initial_training_phase()
    _fill_dict(agent)

    actor_encoder_before = _clone_params(agent.policy.features_extractor)
    critic_encoder_before = _clone_params(agent.policy.critic_features_extractor)

    agent.train(gradient_steps=1, compute_info=True)

    # Actor's own (non-shared) encoder is actor-exclusive: nothing else
    # trains it, so the actor loss must reach it despite RGBD's default
    # detach-on-actor convention (which only applies when the encoder is
    # shared with the critic) -- see SAC._actor_stop_gradient.
    assert _params_changed(actor_encoder_before, agent.policy.features_extractor)
    assert not _params_changed(critic_encoder_before, agent.policy.critic_features_extractor)


def test_shared_encoder_still_detaches_actor_loss_gradient():
    """Regression: without a separate critic extractor, RGBD's existing
    Q-loss-only encoder convention is unchanged."""
    agent = _dict_agent(
        initial_training_phase=InitialTrainingPhase(
            duration_steps=1_000, update_actor=True, update_critic=False, update_encoder=False
        ),
    )
    agent._start_initial_training_phase()
    _fill_dict(agent)

    encoder_before = _clone_params(agent.policy.features_extractor)
    agent.train(gradient_steps=1, compute_info=True)
    assert not _params_changed(encoder_before, agent.policy.features_extractor)


# --- separate critic extractor: PPO gating ---
# Tested at the policy level (forward/evaluate_actions -> backward) rather
# than through a full PPO.learn() rollout, since PPO's single flat optimizer
# means the actual thing to verify is which extractor each loss term's
# gradient reaches -- exactly what evaluate_actions() computes.


def test_ppo_policy_separate_critic_extractor_gates_correctly():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor_extractor = _TrainableBoxExtractor(obs_space, features_dim=6)
    critic_extractor = _TrainableBoxExtractor(obs_space, features_dim=6)
    policy = PPOPolicy(
        observation_space=obs_space,
        action_space=act_space,
        features_extractor=actor_extractor,
        critic_features_extractor=critic_extractor,
    )
    assert policy.critic_features_extractor is not policy.features_extractor

    obs = torch.randn(4, 5)
    actions = torch.randn(4, 2).clamp(-0.9, 0.9)

    values, log_prob, _entropy = policy.evaluate_actions(obs, actions)

    actor_grad = torch.autograd.grad(log_prob.sum(), list(actor_extractor.parameters()), allow_unused=True)
    assert all(g is not None and torch.any(g != 0) for g in actor_grad)
    actor_grad_on_critic = torch.autograd.grad(
        log_prob.sum(), list(critic_extractor.parameters()), retain_graph=True, allow_unused=True
    )
    assert all(g is None for g in actor_grad_on_critic)

    critic_grad = torch.autograd.grad(values.sum(), list(critic_extractor.parameters()), allow_unused=True)
    assert all(g is not None and torch.any(g != 0) for g in critic_grad)


def test_ppo_policy_default_path_shares_extractor():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    extractor = _TrainableBoxExtractor(obs_space, features_dim=6)
    policy = PPOPolicy(observation_space=obs_space, action_space=act_space, features_extractor=extractor)
    assert policy.critic_features_extractor is policy.features_extractor


# --- recurrent policies: explicit unsupported ---


def test_recurrent_sac_policy_rejects_critic_features_extractor():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    extractor = RecordingExtractor(obs_space, features_dim=8)
    recurrent_encoder = RecurrentLatentEncoder(input_dim=8, hidden_size=8)
    with pytest.raises(ValueError, match="critic_features_extractor"):
        RecurrentSACPolicy(
            observation_space=obs_space,
            action_space=act_space,
            features_extractor=extractor,
            recurrent_encoder=recurrent_encoder,
            critic_features_extractor=RecordingExtractor(obs_space, features_dim=8),
        )


def test_recurrent_ppo_policy_rejects_critic_features_extractor():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    extractor = RecordingExtractor(obs_space, features_dim=8)
    recurrent_encoder = RecurrentLatentEncoder(input_dim=8, hidden_size=8)
    with pytest.raises(ValueError, match="critic_features_extractor"):
        RecurrentPPOPolicy(
            observation_space=obs_space,
            action_space=act_space,
            features_extractor=extractor,
            recurrent_encoder=recurrent_encoder,
            critic_features_extractor=RecordingExtractor(obs_space, features_dim=8),
        )


# --- prepare_batch_all: dedup + separate-instance fan-out ---


class _CountingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 5) -> None:
        super().__init__(observation_space, features_dim=features_dim)
        self.calls = 0

    def forward(self, obs):
        batch = next(iter(obs.values())).shape[0] if isinstance(obs, dict) else obs.shape[0]
        return torch.zeros(batch, self.features_dim)

    def prepare_batch(self, obs, next_obs=None) -> None:
        self.calls += 1


def _policy_obs_space() -> spaces.Dict:
    return spaces.Dict({"state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)})


def test_prepare_batch_all_calls_once_when_shared():
    obs_space = _policy_obs_space()
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    extractor = _CountingExtractor(obs_space)
    policy = SACPolicy(observation_space=obs_space, action_space=act_space, features_extractor=extractor)
    policy.prepare_batch_all({"state": torch.zeros(1, 4)})
    assert extractor.calls == 1


def test_prepare_batch_all_calls_both_when_separate():
    obs_space = _policy_obs_space()
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor_extractor = _CountingExtractor(obs_space)
    critic_extractor = _CountingExtractor(obs_space)
    policy = SACPolicy(
        observation_space=obs_space,
        action_space=act_space,
        features_extractor=actor_extractor,
        critic_features_extractor=critic_extractor,
    )
    policy.prepare_batch_all({"state": torch.zeros(1, 4)})
    assert actor_extractor.calls == 1
    assert critic_extractor.calls == 1


def test_combined_extractor_augmentation_cache_keys_are_instance_scoped():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )
    actor_extractor = CombinedExtractor(
        obs_space, image_keys=("rgb",), image_augmentation="random_shift", augmentation_seed=1
    )
    critic_extractor = CombinedExtractor(
        obs_space, image_keys=("rgb",), image_augmentation="random_shift", augmentation_seed=2
    )
    assert actor_extractor._aug_stack_key != critic_extractor._aug_stack_key

    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    actor_extractor.prepare_batch(obs)
    critic_extractor.prepare_batch(obs)
    # Both cached entries must survive in the same obs dict without one
    # clobbering the other.
    assert actor_extractor._aug_stack_key in obs
    assert critic_extractor._aug_stack_key in obs
    assert not torch.equal(obs[actor_extractor._aug_stack_key], obs[critic_extractor._aug_stack_key])


# --- RLPDHybrid: discrete_critic follows the critic role ---


def test_rlpd_hybrid_discrete_critic_sized_from_critic_extractor():
    obs_space = _policy_obs_space()
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    actor_extractor = RecordingExtractor(obs_space, features_dim=13)
    critic_extractor = RecordingExtractor(obs_space, features_dim=9, marker="critic")
    policy = RLPDHybridPolicy(
        observation_space=obs_space,
        action_space=act_space,
        features_extractor=actor_extractor,
        critic_features_extractor=critic_extractor,
    )
    assert policy.discrete_critic.net[0].in_features == 9


# --- separate critic extractor: real CombinedExtractor drops unrequested keys ---


def test_separate_critic_extractor_drops_unrequested_image_key():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0.0, high=1.0, shape=(64, 64, 1), dtype=np.float32),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    agent = SAC(
        env=DummyVecEnv(obs_space, act_space),
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        net_arch={"pi": [16], "qf": [16]},
        image_keys=("rgb", "depth"),
        proprio_latent_dim=4,
        policy_kwargs={
            "critic_features_extractor_class": CombinedExtractor,
            "critic_features_extractor_kwargs": {
                "image_keys": ("rgb",),
                "proprio_latent_dim": 4,
            },
        },
    )

    critic_extractor = agent.policy.critic_features_extractor
    assert isinstance(critic_extractor, CombinedExtractor)
    assert critic_extractor.image_keys == ("rgb",)
    assert "depth" not in critic_extractor.vector_extractors

    for _ in range(8):
        obs = {
            "rgb": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
            "depth": torch.rand(1, 64, 64, 1),
            "state": torch.randn(1, 4),
        }
        next_obs = {
            "rgb": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
            "depth": torch.rand(1, 64, 64, 1),
            "state": torch.randn(1, 4),
        }
        actions = torch.randn(1, *act_space.shape).clamp(-1, 1)
        rewards = torch.randn(1)
        dones = torch.zeros(1)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

    info = agent.train(gradient_steps=1, compute_info=True)
    assert info
