from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import Off2OnFloQ
from rl_garden.buffers.h5_dataset import load_h5_dataset_to_replay_buffer

OBS_DIM = 4
ACTION_DIM = 2

_METRIC_KEYS = (
    "critic_loss",
    "floq_loss",
    "distilled_critic_loss",
    "actor_loss",
    "bc_flow_loss",
    "distill_loss",
    "q_loss",
)


def _write_h5_dataset(path, *, num_traj: int, steps_per_traj: int) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for traj_idx in range(num_traj):
            g = f.create_group(f"traj_{traj_idx}")
            g.create_dataset(
                "obs", data=rng.standard_normal((steps_per_traj + 1, OBS_DIM)).astype(np.float32)
            )
            g.create_dataset(
                "actions", data=(rng.random((steps_per_traj, ACTION_DIM)).astype(np.float32) * 2 - 1)
            )
            g.create_dataset("rewards", data=np.ones(steps_per_traj, dtype=np.float32))
            dones = np.zeros(steps_per_traj, dtype=np.float32)
            dones[-1] = 1.0
            g.create_dataset("dones", data=dones)


class _FakeEnv:
    """SAME_STEP-autoreset fake vector env, fixed episode length. Minimal
    copy of tests/test_acfql_smoke.py's ``_FakeEnv`` (no chunking)."""

    def __init__(self, num_envs: int = 3, episode_len: int = 6) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
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
        done = self._step_count >= self.episode_len
        reward = torch.ones(self.num_envs)
        info = {}
        if done.any():
            info = {
                "final_observation": torch.randn(self.num_envs, OBS_DIM),
                "_final_observation": done.clone(),
                "final_info": {"episode": {"return": self._step_count.float() * reward}},
                "_final_info": done.clone(),
            }
            self._step_count[done] = 0
        obs = torch.randn(self.num_envs, OBS_DIM)
        terminated = done.clone()
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, info


def _make_agent(**overrides) -> Off2OnFloQ:
    kwargs = dict(
        env=_FakeEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=300,
        batch_size=8,
        learning_starts=9,
        training_freq=3,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        noise_samples=2,
        critic_flow_steps=2,
        num_bins=9,
    )
    kwargs.update(overrides)
    return Off2OnFloQ(**kwargs)


def _load_offline(agent: Off2OnFloQ, tmp_path) -> None:
    path = tmp_path / "floq.h5"
    _write_h5_dataset(path, num_traj=6, steps_per_traj=20)
    load_h5_dataset_to_replay_buffer(agent.replay_buffer, str(path))


def test_off2on_floq_defaults_match_floq_recipe():
    agent = _make_agent()
    assert agent.q_agg == "mean"
    assert agent.r_min == -1.0
    assert agent.r_max == 0.0
    assert agent.flow_num_ensembles == 2


def test_off2on_floq_offline_training_produces_finite_losses(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    losses = agent.train(10, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])


def test_off2on_floq_switch_to_online_and_learn(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    agent.train(5)
    agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
    agent.learn(total_timesteps=30)
    assert agent._global_step >= 30
    losses = agent.train(4, compute_info=True)
    for key in _METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])


def test_off2on_floq_checkpoint_roundtrip(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    agent.train(5)
    ckpt = agent.save(tmp_path / "floq.pt")

    agent2 = _make_agent()
    agent2.load(ckpt)
    for (n1, p1), (n2, p2) in zip(
        agent.policy.floq.named_parameters(),
        agent2.policy.floq.named_parameters(),
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
    assert agent2.r_min == agent.r_min
    assert agent2.flow_num_ensembles == agent.flow_num_ensembles
