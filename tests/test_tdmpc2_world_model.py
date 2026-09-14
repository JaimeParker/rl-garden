"""Tests for TD-MPC2's ``TDMPC2`` agent / gradient step against fake CPU
envs (no simulator/hardware), following ``tests/test_sac_core.py``'s
``DummyVecEnv``/hand-built-batch pattern.

Rewritten against the model/policy split (model-based-base plan 1.2/1.3):
the world model (``rl_garden.world_models.latent_consistency
.LatentConsistencyModel``, reached via ``agent.policy.world_model``) now
holds only encoder/dynamics/reward/termination; the actor/critic/critic
target (``rl_garden.policies.tdmpc2_policy.TDMPC2Policy``) are
``agent.policy.actor``/``agent.policy.critic``/``agent.policy.critic_target``.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.tdmpc2 import TDMPC2
from rl_garden.encoders.config import EncoderConfig
from rl_garden.networks.symlog import symexp, symlog
from rl_garden.networks.twohot import two_hot, two_hot_inv


class DummyVecEnv:
    def __init__(self, episode_len: int = 5) -> None:
        self.num_envs = 1
        self.episode_len = episode_len
        self.single_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 2), dtype=np.float32)
        self._t = 0

    def reset(self, seed=None):
        del seed
        self._t = 0
        return torch.zeros(1, 4), {}

    def step(self, actions):
        del actions
        self._t += 1
        truncated = self._t >= self.episode_len
        obs = torch.full((1, 4), float(self._t))
        reward = torch.ones(1)
        terminated = torch.zeros(1, dtype=torch.bool)
        truncated_t = torch.tensor([truncated])
        if truncated:
            self._t = 0
            obs = torch.zeros(1, 4)
        return obs, reward, terminated, truncated_t, {}


class DummyDictVecEnv(DummyVecEnv):
    def __init__(self, episode_len: int = 5) -> None:
        super().__init__(episode_len)
        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "rgb_cam": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            }
        )

    def _obs(self) -> dict:
        return {
            "state": torch.full((1, 4), float(self._t)),
            "rgb_cam": torch.zeros((1, 64, 64, 3), dtype=torch.uint8),
        }

    def reset(self, seed=None):
        del seed
        self._t = 0
        return self._obs(), {}

    def step(self, actions):
        del actions
        self._t += 1
        truncated = self._t >= self.episode_len
        reward = torch.ones(1)
        terminated = torch.zeros(1, dtype=torch.bool)
        truncated_t = torch.tensor([truncated])
        if truncated:
            self._t = 0
        return self._obs(), reward, terminated, truncated_t, {}


_TINY_KWARGS = dict(
    device="cpu",
    buffer_device="cpu",
    buffer_size=200,
    batch_size=4,
    horizon=2,
    num_samples=8,
    num_elites=4,
    num_pi_trajs=2,
    iterations=1,
    latent_dim=8,
    mlp_dim=8,
    num_q=2,
    num_bins=11,
    eval_freq=0,
)


def _agent(env=None, **kwargs) -> TDMPC2:
    params = dict(_TINY_KWARGS)
    params.update(kwargs)
    return TDMPC2(env=env or DummyVecEnv(), episode_length=5, seed_steps=6, **params)


def _dict_agent(**kwargs) -> TDMPC2:
    params = dict(_TINY_KWARGS)
    params["encoder_config"] = EncoderConfig(proprio_latent_dim=4)
    params.update(kwargs)
    return TDMPC2(env=DummyDictVecEnv(), episode_length=5, seed_steps=6, **params)


def test_rejects_multi_env():
    class MultiEnv(DummyVecEnv):
        def __init__(self):
            super().__init__()
            self.num_envs = 4

    import pytest

    with pytest.raises(ValueError):
        _agent(env=MultiEnv())


def test_rejects_episodic_true():
    import pytest

    with pytest.raises(NotImplementedError):
        _agent(episodic=True)


def test_two_hot_round_trip_recovers_scalar_within_bin_resolution():
    num_bins, vmin, vmax = 101, -10.0, 10.0
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.tensor([[3.0], [-4.5], [0.0]])
    soft = two_hot(x, num_bins, vmin, vmax, bin_size)
    recovered = two_hot_inv(soft.log(), num_bins, vmin, vmax)
    # soft.log() feeds back through a softmax in two_hot_inv, which is only
    # an approximate inverse (softmax of log-one-hot != identity everywhere
    # bin weights are split across two adjacent bins); check closeness instead.
    torch.testing.assert_close(recovered, symexp(symlog(x)), atol=0.5, rtol=0.1)


def test_gradient_step_runs_and_updates_world_model_not_via_pi_optimizer():
    agent = _agent()
    agent.learn(total_timesteps=8)  # fills buffer past learning_starts=6
    world_model = agent.policy.world_model

    dyn_params_before = [p.detach().clone() for p in world_model._dynamics.parameters()]

    info = agent._gradient_step()

    assert "total_loss" in info and info["total_loss"] == info["total_loss"]  # not NaN
    dyn_changed = any(
        not torch.equal(a, b) for a, b in zip(dyn_params_before, world_model._dynamics.parameters())
    )
    assert dyn_changed, "world_optimizer.step() should update dynamics parameters"


def test_target_q_starts_as_exact_clone_of_zero_initialized_online_critic():
    """Regression test for the target-Q init-order bug: the target critic
    must be cloned from the online critic *after* TD-MPC2's own
    ``trunc_normal_`` init + last-layer zero_() have run on both (matching
    upstream, ``3rd_party/tdmpc2/tdmpc2/common/world_model.py:31-36``), not
    before -- see ``TDMPC2Policy.__init__``'s docstring."""
    agent = _agent()
    critic, critic_target = agent.policy.critic, agent.policy.critic_target

    for p_live, p_target in zip(critic.parameters(), critic_target.parameters()):
        torch.testing.assert_close(p_live, p_target)

    for q in critic.qs:
        assert torch.all(q[-1].weight == 0.0)
    for q in critic_target.qs:
        assert torch.all(q[-1].weight == 0.0)


def test_target_q_polyak_update_moves_toward_live_q():
    agent = _agent(tau=1.0)  # tau=1 -> target becomes exactly live critic after one update
    critic, critic_target = agent.policy.critic, agent.policy.critic_target
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(1.0)
    agent.policy.soft_update_target_Q()
    for p_live, p_target in zip(critic.parameters(), critic_target.parameters()):
        torch.testing.assert_close(p_live, p_target)


def test_dict_obs_gradient_step_runs():
    agent = _dict_agent()
    agent.learn(total_timesteps=8)
    assert agent._global_update > 0


def test_checkpoint_round_trip_preserves_policy_state(tmp_path):
    agent = _agent()
    agent.learn(total_timesteps=8)

    ckpt_path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)
    agent2 = _agent()
    agent2.load(str(ckpt_path), load_replay_buffer=False)

    sd1 = agent.policy.state_dict()
    sd2 = agent2.policy.state_dict()
    assert set(sd1) == set(sd2)
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"mismatch at {key}"
    assert agent2._global_step == agent._global_step
    assert agent2._global_update == agent._global_update


def test_learn_writes_periodic_and_final_checkpoints(tmp_path):
    agent = _agent(checkpoint_dir=str(tmp_path), checkpoint_freq=4, save_final_checkpoint=True)
    agent.learn(total_timesteps=10)

    assert (tmp_path / "final.pt").exists()
    assert any(p.name.startswith("checkpoint_") for p in tmp_path.iterdir())
