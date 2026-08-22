from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.plas import PLAS


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> PLAS:
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
    return PLAS(**defaults)


def _fill(agent: PLAS, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def _pretrained_agent(**kwargs) -> PLAS:
    agent = _make_agent(vae_iterations=5, **kwargs)
    _fill(agent)
    agent.fit_obs_normalizer()
    agent.pretrain_vae()
    return agent


def test_rejects_unsupported_observation_space():
    unsupported = OfflineEnvSpec(
        spaces.MultiDiscrete([3, 3]),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="Box"):
        PLAS(env=unsupported, buffer_device="cpu", device="cpu")


def test_gradient_step_produces_finite_losses():
    agent = _pretrained_agent()
    metrics = agent.train(4, compute_info=True)
    for key in ("critic_loss", "actor_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_vae_frozen_after_pretrain():
    agent = _pretrained_agent()
    for p in agent.policy.vae.parameters():
        assert p.requires_grad is False
    assert agent.policy.vae.training is False

    before = [p.clone() for p in agent.policy.vae.parameters()]
    agent.train(3)
    after = list(agent.policy.vae.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_pretrain_vae_is_idempotent():
    agent = _make_agent(vae_iterations=5)
    _fill(agent)
    agent.fit_obs_normalizer()
    agent.pretrain_vae()
    weights_after_first = [p.clone() for p in agent.policy.vae.parameters()]
    result = agent.pretrain_vae()
    assert result == {}
    assert all(
        torch.equal(a, b) for a, b in zip(weights_after_first, agent.policy.vae.parameters())
    )


def test_actor_and_critic_targets_update_every_step_no_delay():
    agent = _pretrained_agent()
    latent_actor_target_before = [p.clone() for p in agent.policy.latent_actor_target.parameters()]
    critic_target_before = [p.clone() for p in agent.policy.critic_target.parameters()]

    agent.train(1)

    assert not all(
        torch.equal(a, b)
        for a, b in zip(
            latent_actor_target_before, agent.policy.latent_actor_target.parameters()
        )
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(critic_target_before, agent.policy.critic_target.parameters())
    )


def test_soft_q_lambda_boundaries_drive_train_through_actual_mixture_formula():
    """Drives PLASCore.train() itself (not a standalone tensor expression) --
    same technique as BCQ's analogous test. Stubs policy.q_values so
    target-side (q1, q2) = (10, 2) for every row and live-side (q1, q2) =
    (0, 0). soft_q_lambda=1.0 must select min=2 (critic_loss=8);
    soft_q_lambda=0.0 must select max=10 (critic_loss=200)."""

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
        agent = _pretrained_agent(soft_q_lambda=soft_q_lambda)
        agent.gamma = 1.0
        obs = torch.randn(batch_size, 6)
        next_obs = torch.randn(batch_size, 6)
        actions = torch.rand(batch_size, 3) * 2 - 1
        fake_batch = types.SimpleNamespace(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size),
        )
        agent._sample_train_batch = lambda n: fake_batch
        agent.policy.q_values = fake_q_values

        metrics = agent.train(1, compute_info=True)
        assert metrics["critic_loss"] == pytest.approx(expected_loss, rel=1e-4)


def test_soft_q_lambda_out_of_range_rejected():
    with pytest.raises(ValueError, match="soft_q_lambda"):
        _make_agent(soft_q_lambda=1.5)


def test_use_perturbation_toggle_adds_extra_trainable_parameters():
    agent_plain = _make_agent(use_perturbation=False)
    agent_p = _make_agent(use_perturbation=True)
    plain_actor_params = sum(p.numel() for p in agent_plain.policy.actor_parameters())
    p_actor_params = sum(p.numel() for p in agent_p.policy.actor_parameters())
    assert p_actor_params > plain_actor_params
    assert agent_p.policy.perturbation is not None
    assert agent_plain.policy.perturbation is None


def test_actor_parameters_covers_exactly_latent_actor_and_perturbation_no_targets():
    """Regression guard for the -P organization: actor_parameters() must
    yield every latent_actor + perturbation parameter exactly once, and
    never anything from the frozen *_target copies -- a future refactor
    that mis-wires this would silently break the actor_optimizer's
    coverage without any other test noticing."""
    agent = _make_agent(use_perturbation=True)
    policy = agent.policy

    actor_param_ids = {id(p) for p in policy.actor_parameters()}
    expected_ids = {id(p) for p in policy.latent_actor.parameters()} | {
        id(p) for p in policy.perturbation.parameters()
    }
    assert actor_param_ids == expected_ids

    target_ids = {id(p) for p in policy.latent_actor_target.parameters()} | {
        id(p) for p in policy.perturbation_target.parameters()
    }
    assert actor_param_ids.isdisjoint(target_ids)


def test_perturbation_target_updates_alongside_latent_actor_target():
    """Regression guard: PLASCore.train() must fire BOTH polyak_update calls
    when use_perturbation=True -- catches a future refactor that
    accidentally drops or no-ops one of the two target updates."""
    agent = _pretrained_agent(use_perturbation=True)
    latent_actor_target_before = [
        p.clone() for p in agent.policy.latent_actor_target.parameters()
    ]
    perturbation_target_before = [
        p.clone() for p in agent.policy.perturbation_target.parameters()
    ]

    agent.train(1)

    assert not all(
        torch.equal(a, b)
        for a, b in zip(
            latent_actor_target_before, agent.policy.latent_actor_target.parameters()
        )
    )
    assert not all(
        torch.equal(a, b)
        for a, b in zip(
            perturbation_target_before, agent.policy.perturbation_target.parameters()
        )
    )


def test_checkpoint_round_trip():
    import os
    import tempfile

    agent = _pretrained_agent()
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
    assert loaded._vae_pretrained is True
    for p in loaded.policy.vae.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_smoke():
    agent = _make_agent(buffer_device="cuda", device="cuda", vae_iterations=5)
    _fill(agent, steps=256)
    agent.fit_obs_normalizer()
    agent.pretrain_vae()
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    for p in agent.policy.parameters():
        assert p.grad is None
