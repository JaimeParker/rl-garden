from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import FlowPPO
from rl_garden.algorithms.flow_ppo import FlowPPOCore
from rl_garden.algorithms.on_policy import OnPolicyAlgorithm
from rl_garden.algorithms.ppo import PPO
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.envs.wrappers import ActionChunkWrapper
from rl_garden.networks.actor_vector_field import ActorVectorField, flow_sde_step

OBS_DIM = 5
ACTION_DIM = 2
EPISODE_LEN = 6


class _FakeEnv(gym.Env):
    """SAME_STEP-autoreset fake vector env, fixed episode length -- matches
    ``test_dppo_smoke.py``'s fixture."""

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step_count.zero_()
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, action):
        del action
        self._step_count += 1
        done = self._step_count >= EPISODE_LEN
        reward = torch.ones(self.num_envs)
        final_obs = torch.randn(self.num_envs, OBS_DIM)
        info = {}
        if done.any():
            info = {
                "final_observation": final_obs,
                "_final_observation": done.clone(),
                "final_info": {"episode": {"return": (self._step_count.float() * reward)}},
                "_final_info": done.clone(),
            }
            self._step_count[done] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        terminated = done.clone()
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, info


def _make_env(num_envs: int, horizon_length: int) -> ActionChunkWrapper:
    return ActionChunkWrapper(_FakeEnv(num_envs), act_steps=horizon_length)


def _make_agent(**overrides):
    env = _make_env(num_envs=4, horizon_length=2)
    kwargs = dict(
        env=env,
        num_steps=3,
        horizon_length=2,
        flow_steps=4,
        actor_mlp_dims=[16, 16, 16],
        critic_mlp_dims=[16, 16, 16],
        update_epochs=2,
        update_batch_size=8,
        eval_freq=0,
        device="cpu",
    )
    kwargs.update(overrides)
    return FlowPPO(**kwargs)


def test_mro_has_no_dppo_or_ppo():
    from rl_garden.algorithms.dppo import DPPO, DPPOCore

    mro = FlowPPO.__mro__
    assert mro[:3] == (FlowPPO, FlowPPOCore, OnPolicyAlgorithm)
    assert PPO not in mro
    assert DPPO not in mro
    assert DPPOCore not in mro


def test_registry_discovers_flow_ppo():
    from rl_garden.training.online._registry import registry

    registry.discover()
    assert "flow_ppo" in registry._entries


def test_rejects_unchunked_env():
    env = _FakeEnv(num_envs=2)
    with pytest.raises(ValueError, match="ActionChunkWrapper"):
        FlowPPO(env=env, num_steps=2, horizon_length=3, flow_steps=4, device="cpu")


def test_sde_step_matches_euler_integrate_at_zero_noise_level():
    """Pins the sigma<->tau re-derivation: at noise_level=0, both variants
    must collapse exactly onto ActorVectorField.integrate()'s own Euler
    step, since a sign error in the sigma/tau substitution would break
    this."""
    torch.manual_seed(0)
    net = ActorVectorField(4, 3, [16, 16], use_time_conditioning=True)
    net.eval()
    features = torch.randn(5, 4)
    x = torch.randn(5, 3)
    num_steps = 6
    step = 2
    tau = torch.full((5, 1), step / num_steps)
    dtau = 1.0 / num_steps
    with torch.no_grad():
        v = net(features, x, tau)
        x_ref = x + v / num_steps
        for sde_type in ("sde", "cps"):
            mean, std = flow_sde_step(
                v, x, tau, dtau, sde_type=sde_type, noise_level=0.0, clip_std_min=0.0
            )
            assert torch.allclose(mean, x_ref, atol=1e-6), sde_type
            assert torch.allclose(std, torch.zeros_like(std))


def test_cps_final_step_density_is_guarded_by_clip_std_min():
    """The final SDE step's std_dev_t is exactly 0 for "cps" regardless of
    noise_level -- clip_std_min must guard against a near-zero-std
    division in flow_logprob."""
    torch.manual_seed(0)
    net = ActorVectorField(4, 3, [16, 16], use_time_conditioning=True)
    x = torch.randn(5, 3)
    features = torch.randn(5, 4)
    num_steps = 4
    tau_last = torch.full((5, 1), (num_steps - 1) / num_steps)
    with torch.no_grad():
        v = net(features, x, tau_last)
        _, std_unclipped = flow_sde_step(
            v, x, tau_last, 1.0 / num_steps, sde_type="cps", noise_level=0.7, clip_std_min=0.0
        )
        assert torch.allclose(std_unclipped, torch.zeros_like(std_unclipped))
        _, std_clipped = flow_sde_step(
            v, x, tau_last, 1.0 / num_steps, sde_type="cps", noise_level=0.7, clip_std_min=0.0067
        )
        assert torch.allclose(std_clipped, torch.full_like(std_clipped, 0.0067))


def test_learn_and_train_runs_and_produces_finite_losses():
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)
    losses = agent.train()
    for key, value in losses.items():
        assert np.isfinite(value), (key, value)


def test_old_logprobs_match_recomputed_logprobs_immediately_after_rollout():
    """Pins the rollout<->training (chain-index <-> tau) alignment: DPPO's
    denoising-step indexing runs in reverse (t = arange(k-1,-1,-1)) while
    FlowPPO's runs forward (tau = step/flow_steps) -- a mismatch here would
    silently corrupt every ratio while still producing plausible-looking
    finite losses. Recomputing get_logprobs_subsample right after rollout
    (actor unchanged) must reproduce the stored old_log_probs exactly.
    ``learn()`` itself calls ``train()`` (and so mutates the actor) as soon
    as one rollout window completes, so ``train`` is stubbed out here to
    isolate a pure rollout collection."""
    torch.manual_seed(0)
    agent = _make_agent()
    agent.train = lambda: {}
    agent.learn(total_timesteps=3 * 4 * 2)

    num_steps, num_envs, k = agent.num_steps, agent.num_envs, agent.flow_steps
    total = num_steps * num_envs
    obs_flat = flatten_leading_dims(agent.rollout_buffer.obs)
    chains_flat = agent._chain_buffer.chains.reshape(total, k + 1, agent.policy.action_dim)
    old_logprobs_flat = agent._chain_buffer.old_log_probs.reshape(total, k, agent.policy.action_dim)

    batch_inds = torch.arange(total)
    for step in range(k):
        flow_step_inds = torch.full((total,), step, dtype=torch.long)
        features_b = agent.policy._features(index_obs(obs_flat, batch_inds))
        chains_prev_b = chains_flat[batch_inds, flow_step_inds]
        chains_next_b = chains_flat[batch_inds, flow_step_inds + 1]
        with torch.no_grad():
            recomputed = agent.policy.get_logprobs_subsample(
                features_b, chains_prev_b, chains_next_b, flow_step_inds
            )
        stored = old_logprobs_flat[batch_inds, flow_step_inds]
        assert torch.allclose(recomputed, stored, atol=1e-5), step


def test_train_updates_actor_and_critic():
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)

    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    critic_before = [p.clone() for p in agent.policy.critic.parameters()]

    agent.train()

    assert any(
        not torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(critic_before, agent.policy.critic.parameters())
    )


def test_deterministic_eval_adds_no_flow_step_noise():
    """predict(deterministic=True) still draws a fresh random x_0 (flow
    matching always starts from a prior sample, same as
    ActorVectorField.integrate()'s own contract) -- but must add zero noise
    across the flow steps themselves, unlike stochastic rollout sampling.
    Isolate that by reseeding before each call: same x_0 + zero step-noise
    must give bit-identical actions; stochastic sampling must not."""
    agent = _make_agent()
    obs = torch.randn(4, OBS_DIM)
    with torch.no_grad():
        torch.manual_seed(1)
        det1 = agent.policy.predict(obs, deterministic=True)
        torch.manual_seed(1)
        det2 = agent.policy.predict(obs, deterministic=True)
        # Same seed -> identical x_0 draw, but stochastic mode additionally
        # consumes RNG state (and adds noise) at every flow step, so it
        # must diverge from the deterministic (noise-free) trajectory.
        torch.manual_seed(1)
        stoch1 = agent.policy.predict(obs, deterministic=False)
    assert torch.equal(det1, det2)
    assert not torch.equal(det1, stoch1)
    assert det1.shape == (4, agent.horizon_length, ACTION_DIM)


def test_checkpoint_roundtrip_includes_actor_and_critic(tmp_path):
    torch.manual_seed(0)
    agent = _make_agent()
    agent.learn(total_timesteps=3 * 4 * 2)
    agent.train()

    path = agent.save(str(tmp_path / "ckpt.pt"))
    agent2 = _make_agent()
    agent2.load(path, load_replay_buffer=False)

    for (n1, p1), (n2, p2) in zip(
        agent.policy.actor.named_parameters(), agent2.policy.actor.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
    for (n1, p1), (n2, p2) in zip(
        agent.policy.critic.named_parameters(), agent2.policy.critic.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)
