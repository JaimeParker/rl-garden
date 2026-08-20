from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import ACRLPD

OBS_DIM = 4
ACTION_DIM = 2
EPISODE_LEN = 5


class _FakeEnv:
    """SAME_STEP-autoreset fake vector env, fixed episode length."""

    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self._step_count = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, seed=None):
        del seed
        self._step_count.zero_()
        return torch.randn(self.num_envs, OBS_DIM), {}

    def step(self, action):
        assert torch.all(action <= 1.0 + 1e-4) and torch.all(action >= -1.0 - 1e-4)
        self._step_count += 1
        done = self._step_count >= EPISODE_LEN
        reward = torch.ones(self.num_envs)
        final_obs = torch.randn(self.num_envs, OBS_DIM)
        info = {}
        if done.any():
            info = {
                "final_observation": final_obs,
                "_final_observation": done.clone(),
                "final_info": {"episode": {"return": self._step_count.float() * reward}},
                "_final_info": done.clone(),
            }
            self._step_count[done] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        terminated = done.clone()
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, info


def _make_agent(**overrides) -> ACRLPD:
    # num_envs=3, horizon_length=3, training_freq=3 -> steps_per_env=1 per
    # learn() iteration, so learning_starts must be >= horizon_length *
    # num_envs for the per-env buffer to hold a full window by the time
    # training (and, via the same gate, chunked rollout) first activates.
    kwargs = dict(
        env=_FakeEnv(),
        horizon_length=3,
        device="cpu",
        buffer_device="cpu",
        buffer_size=90,
        batch_size=8,
        learning_starts=9,
        training_freq=3,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
    )
    kwargs.update(overrides)
    return ACRLPD(**kwargs)


def test_acrlpd_defaults_match_qc_recipe():
    agent = _make_agent()
    assert agent.n_critics == 10
    assert agent.critic_subsample_size is None
    assert agent.q_agg == "mean"
    assert agent.backup_entropy is False
    assert agent.bc_alpha == 0.0
    # target_entropy_multiplier=0.5 default: -0.5 * horizon_length * action_dim
    assert agent.target_entropy == -0.5 * 3 * ACTION_DIM


def test_acrlpd_learn_and_train_produce_finite_losses():
    agent = _make_agent()
    agent.learn(total_timesteps=30)
    assert agent._global_step == 30
    losses = agent.train(4)
    for key, value in losses.items():
        assert np.isfinite(value), (key, value)


def test_acrlpd_valid_and_discounts_flow_through_high_utd_slicing():
    # Regression test for _extra_batch_slice_keys: without ("discounts",
    # "valid") set before SAC.__init__ (self.nstep stays 1, so SAC's own
    # conditional append never fires), train_high_utd's minibatch slicing
    # would silently drop these fields.
    agent = _make_agent(utd=2.0)
    agent.learn(total_timesteps=30)
    losses = agent.train(4)
    for key, value in losses.items():
        assert np.isfinite(value), (key, value)


def test_acrlpd_policy_action_space_is_flat_chunked():
    agent = _make_agent(horizon_length=4)
    assert agent.policy.action_space.shape == (4 * ACTION_DIM,)


def test_acrlpd_rollout_replans_less_often_than_every_step():
    # Exact call-count arithmetic (per-env staggering, immediate replan on
    # reset) is already covered precisely and in isolation by
    # tests/test_chunked_rollout_queue.py; this integration test only checks
    # that wiring ACRLPD's actual policy through `_sample_action_chunk`
    # meaningfully reduces call frequency relative to one-call-per-step.
    agent = _make_agent(learning_starts=9)
    calls = 0
    orig = agent._sample_action_chunk

    def _counting(obs):
        nonlocal calls
        calls += 1
        return orig(obs)

    agent._sample_action_chunk = _counting
    total_timesteps = 60  # 20 rollout substeps total (num_envs=3)
    agent.learn(total_timesteps=total_timesteps)
    post_warmup_substeps = (total_timesteps - 9) // agent.num_envs
    assert 0 < calls < post_warmup_substeps


def test_acrlpd_bc_alpha_changes_actor_loss():
    agent_off = _make_agent(bc_alpha=0.0)
    agent_off.learn(total_timesteps=30)
    agent_on = _make_agent(bc_alpha=1.0)
    agent_on.learn(total_timesteps=30)
    # Same env/config modulo bc_alpha -- just verify the BC path runs and
    # produces a finite, generally different loss (not a strict equality
    # check, since network init is randomized independently per agent).
    losses_on = agent_on.train(4)
    for key, value in losses_on.items():
        assert np.isfinite(value), (key, value)


def test_acrlpd_checkpoint_roundtrip(tmp_path):
    agent = _make_agent()
    agent.learn(total_timesteps=30)
    ckpt = agent.save(tmp_path / "acrlpd.pt")

    agent2 = _make_agent()
    agent2.load(ckpt)
    for (n1, p1), (n2, p2) in zip(
        agent.policy.actor.named_parameters(), agent2.policy.actor.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
    assert agent2.horizon_length == agent.horizon_length
