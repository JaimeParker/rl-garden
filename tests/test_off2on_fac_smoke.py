from __future__ import annotations

import h5py
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms import Off2OnFAC
from rl_garden.buffers.h5_dataset import load_h5_dataset_to_replay_buffer

OBS_DIM = 4
ACTION_DIM = 2

_AC_METRIC_KEYS = (
    "critic_loss",
    "td_loss",
    "critic_penalty",
    "est_logpi_mean",
    "est_logpi_max",
    "est_logpi_min",
    "est_logbeta_mean",
    "est_logbeta_max",
    "est_logbeta_min",
    "penalty_weight_mean",
    "actor_loss",
    "distill_loss",
    "q_loss",
)
_ONLINE_METRIC_KEYS = _AC_METRIC_KEYS + ("bc_flow_loss",)


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
    copy of tests/test_off2on_fino_smoke.py's ``_FakeEnv`` (no chunking)."""

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


def _make_agent(**overrides) -> Off2OnFAC:
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
        flow_steps=2,
        bc_pretrain_steps=1,
        bc_batch_size=8,
    )
    kwargs.update(overrides)
    return Off2OnFAC(**kwargs)


def _load_offline(agent: Off2OnFAC, tmp_path) -> None:
    path = tmp_path / "fac.h5"
    _write_h5_dataset(path, num_traj=6, steps_per_traj=20)
    load_h5_dataset_to_replay_buffer(agent.replay_buffer, str(path))


def test_off2on_fac_defaults_match_fac_recipe():
    agent = _make_agent()
    assert agent.fac_alpha == 1.0
    assert agent.fac_lambda == 1.0
    assert agent.fac_threshold == "batch_adaptive"
    assert agent._online_finetuning is False


def test_off2on_fac_offline_training_produces_finite_losses(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    losses = agent.train(10, compute_info=True)
    for key in _AC_METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])
    assert agent.phase == "actor_critic"
    assert agent._logp_table is not None


def test_off2on_fac_switch_to_online_and_learn(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    agent.train(5)
    assert agent._online_finetuning is False

    agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
    assert agent._online_finetuning is True

    agent.learn(total_timesteps=30)
    assert agent._global_step >= 30

    losses = agent.train(4, compute_info=True)
    for key in _ONLINE_METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])


def test_off2on_fac_online_step_skips_prelude_and_table():
    # bc_pretrain_steps deliberately large so the online path would fail
    # loudly (buffer underrun / stale table) if train() ever tried to run
    # the offline prelude or dataset-logp pass after the switch.
    agent = _make_agent(bc_pretrain_steps=1_000_000)
    agent.switch_to_online_mode(online_replay_mode="empty")
    assert agent._online_finetuning is True
    # agent.env is boundary-normalized to Dict({"state": Box}) by
    # BaseAlgorithm.__init__ (VectorizedDictStateWrapper), so reset()/step()
    # return {"state": tensor}, not a raw tensor -- see tests/test_fino_core.py::_fill.
    obs, _ = agent.env.reset(seed=0)
    actions = torch.rand(agent.env.num_envs, ACTION_DIM) * 2 - 1
    next_obs = {"state": torch.randn_like(obs["state"])}
    rewards = torch.ones(agent.env.num_envs)
    dones = torch.zeros(agent.env.num_envs)
    for _ in range(agent.batch_size):
        agent.replay_buffer.add(obs, next_obs, actions, rewards, dones)

    losses = agent.train(2, compute_info=True)
    # Load-bearing: if train() had run the (huge) offline prelude instead,
    # this call would hang/time out rather than reach these asserts.
    assert agent._logp_table is None
    assert agent._phase_step == 0
    for key in _ONLINE_METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])


def test_off2on_fac_checkpoint_roundtrip_preserves_online_flag(tmp_path):
    agent = _make_agent()
    _load_offline(agent, tmp_path)
    agent.train(5)
    agent.switch_to_online_mode(online_replay_mode="mixed", offline_data_ratio=0.5)
    agent.train(2)
    ckpt = agent.save(tmp_path / "fac.pt")

    agent2 = _make_agent()
    agent2.load(ckpt)
    assert agent2._online_finetuning is True
    for (n1, p1), (n2, p2) in zip(
        agent.policy.actor_bc_flow.named_parameters(),
        agent2.policy.actor_bc_flow.named_parameters(),
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
    for (n1, p1), (n2, p2) in zip(
        agent.policy.actor_onestep_flow.named_parameters(),
        agent2.policy.actor_onestep_flow.named_parameters(),
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
    for (n1, p1), (n2, p2) in zip(
        agent.policy.critic.named_parameters(),
        agent2.policy.critic.named_parameters(),
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)

    # _logp_table itself DOES round-trip (FACCore._extra_checkpoint_state
    # persists it unconditionally, built once during the offline phase) --
    # what must NOT happen is train() touching it again post-load.
    assert agent2._logp_table is not None
    assert torch.allclose(
        agent2._logp_table.nan_to_num(), agent._logp_table.nan_to_num(), equal_nan=True
    )

    def _must_not_recompute():
        raise AssertionError("_compute_dataset_logp must not run for an online-resumed agent")

    agent2._compute_dataset_logp = _must_not_recompute

    # The reloaded agent must continue in online mode without ever gathering
    # from (or rebuilding) that frozen table -- give it fresh online replay
    # data first, matching what OffPolicyAlgorithm.learn() would add.
    obs, _ = agent2.env.reset(seed=0)
    actions = torch.rand(agent2.env.num_envs, ACTION_DIM) * 2 - 1
    next_obs = {"state": torch.randn_like(obs["state"])}
    rewards = torch.ones(agent2.env.num_envs)
    dones = torch.zeros(agent2.env.num_envs)
    for _ in range(agent2.batch_size):
        agent2.replay_buffer.add(obs, next_obs, actions, rewards, dones)

    table_before = agent2._logp_table.clone()
    losses = agent2.train(2, compute_info=True)
    assert torch.allclose(agent2._logp_table.nan_to_num(), table_before.nan_to_num(), equal_nan=True)
    for key in _ONLINE_METRIC_KEYS:
        assert key in losses
        assert np.isfinite(losses[key]), (key, losses[key])
