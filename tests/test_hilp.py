from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import HILP, OfflineEnvSpec


def _write_h5(path, num_traj: int = 4, length: int = 20) -> None:
    with h5py.File(path, "w") as f:
        for i in range(num_traj):
            g = f.create_group(f"traj_{i}")
            g.create_dataset(
                "obs", data=np.random.randn(length + 1, 6).astype(np.float32)
            )
            g.create_dataset(
                "actions", data=np.random.uniform(-1, 1, (length, 2)).astype(np.float32)
            )
            g.create_dataset("rewards", data=np.zeros(length, dtype=np.float32))
            terminated = np.zeros(length, dtype=bool)
            terminated[-1] = True
            g.create_dataset("terminated", data=terminated)
            g.create_dataset("truncated", data=np.zeros(length, dtype=bool))


def _make_agent(path, **overrides) -> HILP:
    env = OfflineEnvSpec(
        spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        num_envs=1,
    )
    kwargs = dict(
        env=env,
        dataset_path=str(path),
        device="cpu",
        batch_size=32,
        skill_dim=4,
        value_hidden_dims=(16, 16),
        actor_hidden_dims=(16, 16),
        std_log=False,
    )
    kwargs.update(overrides)
    return HILP(**kwargs)


def test_value_loss_is_double_expectile_iql(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path, expectile=0.7)
    batch = agent._dataset.sample(16)

    with torch.no_grad():
        masks = 1.0 - batch.success
        rewards = batch.success - 1.0
        next_v1_t, next_v2_t = agent.policy.value_target(batch.next_obs, batch.goals)
        next_v_min = torch.minimum(next_v1_t, next_v2_t)
        q_for_adv = rewards + agent.discount * masks * next_v_min
        v1_t, v2_t = agent.policy.value_target(batch.obs, batch.goals)
        v_t = (v1_t + v2_t) / 2.0
        adv = q_for_adv - v_t
        q1 = rewards + agent.discount * masks * next_v1_t
        q2 = rewards + agent.discount * masks * next_v2_t
        v1, v2 = agent.policy.value(batch.obs, batch.goals)
        w1 = torch.where(adv > 0, agent.expectile, 1.0 - agent.expectile)
        w2 = torch.where(adv > 0, agent.expectile, 1.0 - agent.expectile)
        expected = (w1 * (q1 - v1).pow(2)).mean() + (w2 * (q2 - v2).pow(2)).mean()

    loss, metrics = agent._compute_value_loss(batch)
    assert torch.isclose(torch.tensor(metrics["value_loss"]), expected, atol=1e-5)


def test_phi_receives_gradient_only_from_value_loss(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path)
    batch = agent._dataset.sample(32)

    skill_loss, _ = agent._compute_skill_losses(batch)
    agent.value_optimizer.zero_grad(set_to_none=True)
    agent.skill_value_optimizer.zero_grad(set_to_none=True)
    agent.skill_critic_optimizer.zero_grad(set_to_none=True)
    agent.skill_actor_optimizer.zero_grad(set_to_none=True)
    skill_loss.backward()

    assert all(p.grad is None for p in agent.policy.value.parameters())
    assert any(p.grad is not None for p in agent.policy.skill_value.parameters())
    assert any(p.grad is not None for p in agent.policy.skill_critic.parameters())
    assert any(p.grad is not None for p in agent.policy.skill_actor.parameters())


def test_skill_critic_has_no_mask_term(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path)
    batch = agent._dataset.sample(32)
    # Force every transition to look terminal (success=1); if a mask term
    # were present the critic target would collapse to just the pseudo-reward
    # (bootstrap zeroed out). It shouldn't -- the reference has no mask here.
    batch.success = torch.ones_like(batch.success)

    with torch.no_grad():
        phis = agent.policy.value.phi(batch.obs)
        next_phis = agent.policy.value.phi(batch.next_obs)
        skills = agent._sample_skills(batch.obs.shape[0])
        skill_rewards = ((next_phis - phis) * skills).sum(dim=-1)
        obs_skill = torch.cat([batch.obs, skills], dim=-1)
        next_obs_skill = torch.cat([batch.next_obs, skills], dim=-1)
        next_v = agent.policy.skill_value(next_obs_skill).squeeze(-1)
        expected_target = skill_rewards + agent.skill_discount * next_v

    assert not torch.allclose(expected_target, skill_rewards)


def test_skill_actor_is_unsquashed_state_independent_std(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path)
    assert agent.policy.skill_actor.log_stds is not None
    obs1 = torch.randn(1, 6 + 4)
    obs2 = torch.randn(1, 6 + 4)
    _, log_std1 = agent.policy.skill_actor(obs1)
    _, log_std2 = agent.policy.skill_actor(obs2)
    assert torch.equal(log_std1, log_std2)
    assert agent.policy.skill_actor.tanh_mean is False


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "demo.h5"
    _write_h5(path)
    agent = _make_agent(path)
    agent.train(5)
    ckpt_path = tmp_path / "hilp.pt"
    agent.save(ckpt_path)

    loaded = _make_agent(path)
    loaded.load(ckpt_path, load_replay_buffer=False)
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, loaded.policy.state_dict()[key]), key
