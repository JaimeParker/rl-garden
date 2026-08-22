from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.bcq import BCQ


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> BCQ:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
        vae_hidden_dim=16,
    )
    defaults.update(kwargs)
    return BCQ(**defaults)


def _fill(agent: BCQ, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="Box"):
        BCQ(env=unsupported, buffer_device="cpu", device="cpu")


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    metrics = agent.train(4, compute_info=True)
    for key in ("critic_loss", "actor_loss", "vae_loss", "vae_recon_loss", "vae_kl_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_vae_is_never_frozen():
    """Regression guard: BCQ's VAE trains jointly every step, unlike SPOT's
    pretrain-then-freeze VAE. It must never end up with requires_grad=False."""
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    for p in agent.policy.vae.parameters():
        assert p.requires_grad is True
    agent.train(3)
    for p in agent.policy.vae.parameters():
        assert p.requires_grad is True
    assert agent.policy.vae.training is True


def test_vae_weights_actually_move_every_step():
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    before = [p.clone() for p in agent.policy.vae.parameters()]
    agent.train(1)
    after = list(agent.policy.vae.parameters())
    assert not all(torch.equal(b, a) for b, a in zip(before, after))


def test_actor_and_targets_update_every_step_no_policy_freq_delay():
    """BCQ has no policy_freq-style delay: actor and both target networks
    update on every single gradient step."""
    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    actor_before = [p.clone() for p in agent.policy.actor.parameters()]
    actor_target_before = [p.clone() for p in agent.policy.actor_target.parameters()]
    critic_target_before = [p.clone() for p in agent.policy.critic_target.parameters()]

    agent.train(1)

    assert not all(
        torch.equal(a, b) for a, b in zip(actor_before, agent.policy.actor.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(actor_target_before, agent.policy.actor_target.parameters())
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(critic_target_before, agent.policy.critic_target.parameters())
    )


def _fake_batch(batch_size: int, obs_dim: int, act_dim: int, *, next_obs_state_id: bool = False):
    obs = torch.randn(batch_size, obs_dim)
    if next_obs_state_id:
        next_obs = torch.zeros(batch_size, obs_dim)
        next_obs[:, 0] = torch.arange(batch_size, dtype=torch.float32)
    else:
        next_obs = torch.randn(batch_size, obs_dim)
    actions = torch.rand(batch_size, act_dim) * 2 - 1
    rewards = torch.zeros(batch_size)
    dones = torch.zeros(batch_size)
    return types.SimpleNamespace(
        obs=obs, next_obs=next_obs, actions=actions, rewards=rewards, dones=dones
    )


def test_soft_q_lambda_boundaries_drive_train_through_actual_mixture_formula():
    """Drives BCQCore.train() itself (not a standalone tensor expression) by
    stubbing policy.q_values so target-side (q1, q2) = (10, 2) for every
    candidate and live-side (q1, q2) = (0, 0). soft_q_lambda=1.0 must select
    min=2 (critic_loss=2*mean(2**2)=8); soft_q_lambda=0.0 must select
    max=10 (critic_loss=2*mean(10**2)=200). A sign/order flip in the actual
    mixture formula inside train() would fail this; a local-tensor
    tautology would not."""

    def fake_q_values(features, actions, target=False):
        del actions
        n = features.shape[0]
        if target:
            return torch.full((n, 1), 10.0), torch.full((n, 1), 2.0)
        # requires_grad=True: this feeds critic_loss/actor_loss outside any
        # no_grad block, so .backward() needs a real graph to walk even
        # though the stub deliberately makes the values themselves zero.
        return (
            torch.zeros(n, 1, requires_grad=True),
            torch.zeros(n, 1, requires_grad=True),
        )

    batch_size = 4
    for soft_q_lambda, expected_loss in ((1.0, 8.0), (0.0, 200.0)):
        agent = _make_agent(soft_q_lambda=soft_q_lambda)
        agent.gamma = 1.0
        agent._sample_train_batch = lambda n: _fake_batch(batch_size, 6, 3)
        agent.policy.q_values = fake_q_values

        metrics = agent.train(1, compute_info=True)
        assert metrics["critic_loss"] == pytest.approx(expected_loss, rel=1e-4)


def test_soft_q_lambda_out_of_range_rejected():
    with pytest.raises(ValueError, match="soft_q_lambda"):
        _make_agent(soft_q_lambda=1.5)


def test_target_repeat_interleave_grouping_matches_reshape_max_in_actual_train():
    """Drives BCQCore.train() itself. Encodes each original next-state's
    identity into next_obs[:, 0] (extract_features is identity-preserving
    here: default obs_mean=0/obs_std=1, FlattenExtractor is a pass-through
    for Box obs), stubs policy.q_values(target=True) to return that identity
    as q1=q2, and sets gamma=1/rewards=0/dones=0 so critic_loss reduces to
    2*mean(mixed_q**2). If BCQCore.train() correctly repeat_interleave-tiles
    next_features into blocks of _NUM_TARGET_CANDIDATES before
    reshape(-1, N).max(1), mixed_q must recover exactly [0, 1, ..., B-1] --
    the clean 0..B-1 pattern would NOT appear if repeat_interleave were
    swapped for plain .repeat (which interleaves different states within
    each reshape block instead of tiling one state N times)."""

    def fake_q_values(features, actions, target=False):
        del actions
        n = features.shape[0]
        if target:
            state_id = features[:, 0].round().unsqueeze(-1)
            return state_id, state_id
        # requires_grad=True: this feeds critic_loss/actor_loss outside any
        # no_grad block, so .backward() needs a real graph to walk even
        # though the stub deliberately makes the values themselves zero.
        return (
            torch.zeros(n, 1, requires_grad=True),
            torch.zeros(n, 1, requires_grad=True),
        )

    batch_size = 4
    agent = _make_agent()
    agent.gamma = 1.0
    agent._sample_train_batch = lambda n: _fake_batch(
        batch_size, 6, 3, next_obs_state_id=True
    )
    agent.policy.q_values = fake_q_values

    metrics = agent.train(1, compute_info=True)
    expected_mixed = torch.arange(batch_size, dtype=torch.float32)
    expected_critic_loss = 2.0 * (expected_mixed**2).mean().item()
    assert metrics["critic_loss"] == pytest.approx(expected_critic_loss, rel=1e-4)


def test_checkpoint_round_trip_preserves_trainable_vae():
    import os
    import tempfile

    agent = _make_agent()
    _fill(agent)
    agent.fit_obs_normalizer()
    agent.train(3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
    # BCQ's VAE must never be frozen, including after a checkpoint load --
    # unlike SPOT/PLAS, there is no pretrained-VAE state to restore.
    assert all(p.requires_grad for p in loaded.policy.vae.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_smoke():
    agent = _make_agent(buffer_device="cuda", device="cuda")
    _fill(agent, steps=256)
    agent.fit_obs_normalizer()
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    for p in agent.policy.parameters():
        assert p.grad is None
