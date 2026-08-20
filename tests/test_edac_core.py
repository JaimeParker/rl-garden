from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import OfflineEnvSpec
from rl_garden.algorithms.edac import EDAC


def _state_env(num_envs: int = 1) -> OfflineEnvSpec:
    return OfflineEnvSpec(
        spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        num_envs=num_envs,
    )


def _make_agent(**kwargs) -> EDAC:
    defaults = dict(
        env=_state_env(),
        buffer_size=1000,
        buffer_device="cpu",
        batch_size=32,
        device="cpu",
        net_arch=[16, 16],
        n_critics=4,
    )
    defaults.update(kwargs)
    return EDAC(**defaults)


def _fill(agent: EDAC, steps: int = 64) -> None:
    env = agent.env
    for _ in range(steps):
        obs = torch.randn(env.num_envs, *env.single_observation_space.shape)
        next_obs = torch.randn_like(obs)
        actions = torch.rand(env.num_envs, *env.single_action_space.shape) * 2 - 1
        rewards = torch.randn(env.num_envs)
        dones = torch.zeros(env.num_envs)
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)


def test_default_n_critics_is_ten():
    agent = _make_agent()
    assert agent.n_critics == 4  # overridden by _make_agent's defaults

    default_agent = EDAC(env=_state_env(), buffer_device="cpu", device="cpu")
    assert default_agent.n_critics == 10


def test_backup_entropy_is_set_true():
    """OfflineSAC.__init__ never sets self.backup_entropy (a pre-existing
    gap); EDAC must provide it itself so SACCore._target_q doesn't crash."""
    agent = _make_agent()
    assert agent.backup_entropy is True


def test_gradient_step_produces_finite_losses():
    agent = _make_agent()
    _fill(agent)
    metrics = agent.train(2, compute_info=True)
    for key in ("critic_loss", "td_loss", "diversity_loss", "actor_loss"):
        assert key in metrics
        assert np.isfinite(metrics[key]), (key, metrics[key])


def test_predict_in_bounds():
    agent = _make_agent()
    obs = torch.randn(4, 6)
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_eta_zero_reduces_critic_loss_to_plain_td_loss():
    """A clean, decisive formula reduction: eta=0 zeroes the diversity term
    entirely, so critic_loss must equal td_loss exactly."""
    agent = _make_agent(eta=0.0)
    _fill(agent)
    data = agent._sample_train_batch(agent.batch_size)
    critic_loss, info = agent._critic_loss(data)
    assert torch.allclose(critic_loss, info["td_loss"])
    assert info["diversity_loss"].item() != 0.0  # still computed, just unweighted


def test_diversity_loss_is_lower_for_a_diverse_ensemble_than_a_collapsed_one():
    """Sanity check on the diversity_loss's direction: an ensemble whose
    critics compute IDENTICAL functions (perfectly correlated action
    gradients) should score a higher (worse) diversity_loss than one with
    independently-initialized critics."""
    agent = _make_agent(n_critics=4)
    _fill(agent)
    data = agent._sample_train_batch(agent.batch_size)
    diverse_loss = agent._critic_diversity_loss(data).item()

    # Collapse the ensemble: copy critic 0's parameters into every other
    # critic, so every g_i is identical -> maximal pairwise cosine similarity.
    import copy

    from rl_garden.networks.actor_critic import EnsembleQCritic

    collapsed_agent = _make_agent(n_critics=4)
    collapsed_agent.replay_buffer = agent.replay_buffer
    src_state = {
        k: v for k, v in agent.policy.critic.state_dict().items()
    }
    # Broadcast critic 0's per-critic params to all critics (vmap "ens_p_"
    # params have a leading n_critics dim).
    collapsed_state = copy.deepcopy(src_state)
    for k, v in collapsed_state.items():
        if v.dim() >= 1 and v.shape[0] == agent.n_critics:
            collapsed_state[k] = v[0:1].expand_as(v).clone()
    collapsed_agent.policy.critic.load_state_dict(collapsed_state)

    collapsed_loss = collapsed_agent._critic_diversity_loss(data).item()
    assert collapsed_loss > diverse_loss


def test_checkpoint_round_trip():
    import os
    import tempfile

    agent = _make_agent()
    _fill(agent)
    for _ in range(3):
        agent.train(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        agent.save(path, include_replay_buffer=False)

        loaded = _make_agent()
        loaded.load(path, load_replay_buffer=False)

    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_smoke():
    agent = _make_agent(buffer_device="cuda", device="cuda")
    _fill(agent, steps=256)
    metrics = agent.train(5, compute_info=True)
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    agent.policy.zero_grad(set_to_none=True)
    obs = torch.randn(4, 6, device="cuda")
    action = agent.policy.predict(obs)
    assert action.shape == (4, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    for p in agent.policy.parameters():
        assert p.grad is None
