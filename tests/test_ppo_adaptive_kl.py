"""Tests for lr_schedule="adaptive_kl": analytic Gaussian KL-driven LR
scheduling, faithfully replicating 3rd_party/rsl_rl's algorithms/ppo.py
adaptive-LR branch, for both plain PPO and RecurrentPPO.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.ppo import PPO
from rl_garden.algorithms.recurrent_ppo import RecurrentPPO
from rl_garden.networks.actor_critic import gaussian_kl_divergence


class _FakeBoxEnv:
    def __init__(self, num_envs: int = 4, episode_len: int = 6, obs_dim: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.obs_dim = obs_dim
        self._t = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _obs(self):
        return torch.randn(self.num_envs, self.obs_dim)

    def reset(self, seed=None):
        del seed
        self._t.zero_()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._t[terminated] = 0
        return self._obs(), reward, terminated, truncated, {}


def _ppo_kwargs(**overrides) -> dict:
    kwargs = dict(
        num_steps=8, num_minibatches=2, update_epochs=2, device="cpu",
        lr_schedule="adaptive_kl", normalize_obs=False, eval_freq=0, log_freq=0,
        net_arch=[16], target_kl=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_gaussian_kl_divergence_matches_hand_computed_reference():
    old_mean = torch.tensor([[0.0, 1.0]])
    old_log_std = torch.tensor([[0.0, 0.0]])  # std = 1
    new_mean = torch.tensor([[1.0, 1.0]])
    new_log_std = torch.tensor([[0.0, 0.0]])  # std = 1

    kl = gaussian_kl_divergence(old_mean, old_log_std, new_mean, new_log_std)
    # KL(N(0,1) || N(1,1)) = 0.5 for the first dim, 0 for the identical second
    # dim (see e.g. https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions).
    assert torch.allclose(kl, torch.tensor([0.5]), atol=1e-5)

    # Identical distributions -> zero KL.
    kl_zero = gaussian_kl_divergence(old_mean, old_log_std, old_mean, old_log_std)
    assert torch.allclose(kl_zero, torch.zeros(1), atol=1e-6)


def test_ppo_adaptive_kl_rollout_mean_matches_stored_buffer_provenance():
    """Pins that the stashed self._rollout_mean/_rollout_log_std correspond
    to the exact obs stored at the same buffer index, not an adjacent step."""
    env = _FakeBoxEnv(num_envs=3)
    agent = PPO(env, **_ppo_kwargs(num_steps=6))
    agent.policy.eval()  # frozen policy: deterministic mean per obs

    obs, _ = env.reset(seed=agent.seed)
    hidden = agent._initial_hidden_state(agent.num_envs)
    next_done = torch.zeros(agent.num_envs, device=agent.device)
    agent.rollout_buffer.reset()
    stored_obs = []
    for _ in range(6):
        actions, values, log_probs, _, hidden = agent._rollout_step(obs, hidden, next_done)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        next_done = torch.logical_or(terminations, truncations).float()
        final_values = agent._compute_final_values(infos, next_done.bool(), hidden)
        agent.rollout_buffer.add(
            obs, actions, rewards, next_done, values, log_probs,
            final_values=final_values, **agent._extra_rollout_buffer_kwargs(),
        )
        stored_obs.append(obs)
        obs = next_obs

    for t in range(6):
        with torch.no_grad():
            expected_mean = agent.policy.actor(
                agent.policy.extract_features(stored_obs[t])
            ).mean
        assert torch.allclose(agent.rollout_buffer.means[t], expected_mean, atol=1e-6)


def test_ppo_adaptive_kl_lr_grows_for_near_unchanged_policy():
    env = _FakeBoxEnv()
    agent = PPO(env, **_ppo_kwargs(desired_kl=0.01))
    lr_before = agent.policy_optimizer.param_groups[0]["lr"]
    agent.learn(total_timesteps=8 * env.num_envs * 2)
    lr_after = agent.policy_optimizer.param_groups[0]["lr"]
    # A freshly-initialized, low-lr policy barely moves per minibatch step ->
    # kl_mean stays well under desired_kl/2 -> LR should grow (clamped at
    # adaptive_lr_max).
    assert lr_after > lr_before


def test_ppo_adaptive_kl_lr_shrinks_for_large_kl():
    env = _FakeBoxEnv()
    agent = PPO(env, **_ppo_kwargs(desired_kl=0.01))
    # Real minibatch evaluate (so values/log_prob/entropy carry a live grad
    # graph through self.policy), but with new_mean overridden to a value far
    # from old_mean to deterministically force the shrink branch regardless
    # of how close the freshly-initialized policy's own KL happens to be.
    agent.rollout_buffer.reset()
    obs, _ = env.reset(seed=agent.seed)
    hidden = agent._initial_hidden_state(agent.num_envs)
    next_done = torch.zeros(agent.num_envs, device=agent.device)
    for _ in range(agent.num_steps):
        actions, values, log_probs, _, hidden = agent._rollout_step(obs, hidden, next_done)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        next_done = torch.logical_or(terminations, truncations).float()
        final_values = agent._compute_final_values(infos, next_done.bool(), hidden)
        agent.rollout_buffer.add(
            obs, actions, rewards, next_done, values, log_probs,
            final_values=final_values, **agent._extra_rollout_buffer_kwargs(),
        )
        obs = next_obs
    with torch.no_grad():
        last_values = agent._predict_last_values(agent._obs_to_policy_device(obs), hidden)
    agent.rollout_buffer.compute_returns_and_advantage(last_values, next_done)

    data = next(agent._iter_minibatches())
    (values, log_prob, entropy, old_values, old_log_prob, advantages, returns,
     old_mean, old_log_std, new_mean, new_log_std) = agent._evaluate_minibatch(data)
    new_mean = new_mean + 10.0  # force a large KL vs. old_mean

    lr_before = agent.policy_optimizer.param_groups[0]["lr"]
    agent._ppo_minibatch_update(
        values=values, log_prob=log_prob, entropy=entropy,
        old_values=old_values, old_log_prob=old_log_prob,
        advantages=advantages, returns=returns, clip_coef=0.2,
        old_mean=old_mean, old_log_std=old_log_std,
        new_mean=new_mean, new_log_std=new_log_std,
    )
    lr_after = agent.policy_optimizer.param_groups[0]["lr"]
    assert lr_after < lr_before


def test_ppo_anneal_lr_with_adaptive_kl_raises():
    env = _FakeBoxEnv()
    with pytest.raises(ValueError, match="anneal_lr"):
        PPO(env, **_ppo_kwargs(anneal_lr=True))


def test_ppo_adaptive_kl_checkpoint_round_trip_preserves_lr(tmp_path):
    env = _FakeBoxEnv()
    agent = PPO(env, **_ppo_kwargs())
    agent.learn(total_timesteps=8 * env.num_envs * 2)
    lr_before = agent.policy_optimizer.param_groups[0]["lr"]

    path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)
    resumed = PPO(env, **_ppo_kwargs())
    resumed.load(path, load_replay_buffer=False)
    assert resumed.policy_optimizer.param_groups[0]["lr"] == lr_before


def test_recurrent_ppo_adaptive_kl_lr_moves():
    env = _FakeBoxEnv()
    agent = RecurrentPPO(env, **_ppo_kwargs())
    lr_before = agent.policy_optimizer.param_groups[0]["lr"]
    agent.learn(total_timesteps=8 * env.num_envs * 2)
    lr_after = agent.policy_optimizer.param_groups[0]["lr"]
    assert lr_after != lr_before


def test_recurrent_ppo_adaptive_kl_rollout_mean_matches_stored_buffer_provenance():
    env = _FakeBoxEnv(num_envs=4)
    agent = RecurrentPPO(env, **_ppo_kwargs(num_steps=6))
    agent.policy.eval()

    obs, _ = env.reset(seed=agent.seed)
    hidden = agent._initial_hidden_state(agent.num_envs)
    next_done = torch.zeros(agent.num_envs, device=agent.device)
    agent.rollout_buffer.reset()
    stored_obs = []
    stored_hidden = []
    for _ in range(6):
        stored_hidden.append(hidden)
        actions, values, log_probs, _, hidden = agent._rollout_step(obs, hidden, next_done)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        next_done = torch.logical_or(terminations, truncations).float()
        final_values = agent._compute_final_values(infos, next_done.bool(), hidden)
        agent.rollout_buffer.add(
            obs, actions, rewards, next_done, values, log_probs,
            final_values=final_values, **agent._extra_rollout_buffer_kwargs(),
        )
        stored_obs.append(obs)
        obs = next_obs

    episode_starts = torch.zeros(agent.num_envs)
    for t in range(6):
        with torch.no_grad():
            raw = agent.policy._extract_features(stored_obs[t], stop_gradient=False)
            latent, _ = agent.policy.recurrent_encoder.step(raw, stored_hidden[t], episode_starts)
            expected_mean = agent.policy.actor(latent).mean
        assert torch.allclose(agent.rollout_buffer.means[t], expected_mean, atol=1e-6)


def test_adaptive_kl_reaches_buffer_through_cli_args_entrypoint():
    """lr_schedule="adaptive_kl" must survive PPOArgs -> _ppo_common_kwargs ->
    construct_agent(PPO, ...) unfiltered -- every other test in this file
    calls PPO(...)/RecurrentPPO(...) directly, bypassing the CLI args layer
    entirely, so this is the only test exercising the actual entrypoint."""
    from rl_garden.training.online.ppo import PPOArgs, build_ppo
    from rl_garden.training.online.recurrent_ppo import RecurrentPPOArgs, build_recurrent_ppo

    env = _FakeBoxEnv()
    args = PPOArgs(
        lr_schedule="adaptive_kl", desired_kl=0.02, num_steps=8, num_minibatches=2,
        update_epochs=1, eval_freq=0, log_freq=0, obs_mode="state",
    )
    agent = build_ppo(args, env, None, None, None)
    assert agent.lr_schedule == "adaptive_kl"
    assert agent.desired_kl == 0.02
    assert agent.rollout_buffer.store_dist_params

    r_args = RecurrentPPOArgs(
        lr_schedule="adaptive_kl", desired_kl=0.02, num_steps=8, num_minibatches=2,
        update_epochs=1, eval_freq=0, log_freq=0, obs_mode="state",
    )
    r_agent = build_recurrent_ppo(r_args, env, None, None, None)
    assert r_agent.lr_schedule == "adaptive_kl"
    assert r_agent.rollout_buffer.store_dist_params
