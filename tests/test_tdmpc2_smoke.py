"""Single-task online TD-MPC2 smoke test: a full ``learn()`` loop (through
the ``learning_starts``/``seed_steps`` pretrain burst) against a fake CPU
env, no simulator/hardware -- same ``DummyVecEnv`` pattern as
``tests/test_recurrent_sac.py``/``tests/test_tdmpc2_world_model.py``.
Runtime target: well under 30s on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms.tdmpc2 import TDMPC2


class DummyVecEnv:
    """``num_envs == 1`` fake env (TD-MPC2 has no vectorized rollout)."""

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
    latent_dim=32,
    mlp_dim=32,
    num_q=2,
    num_bins=11,
    eval_freq=0,
)


def _agent(**kwargs) -> TDMPC2:
    params = dict(_TINY_KWARGS)
    params.update(kwargs)
    return TDMPC2(env=DummyVecEnv(), episode_length=5, seed_steps=16, **params)


def test_learn_runs_seed_step_burst_and_produces_finite_losses():
    agent = _agent()
    agent.learn(total_timesteps=64)  # crosses learning_starts=16 -> pretrain burst

    assert agent._global_step == 64
    # seed_steps=16 == learning_starts triggers a pretrain burst of
    # `learning_starts` gradient steps exactly at that boundary, then one
    # more per subsequent env step (see TDMPC2.learn()): 16 (burst) + 48
    # (steps 17..64) == 64. A regression to "one update per step throughout"
    # would under-count this, so assert the exact total, not just > 0.
    assert agent._global_update == 64

    info = agent._gradient_step()
    for key in ("consistency_loss", "reward_loss", "value_loss", "total_loss", "pi_loss"):
        assert key in info
        assert info[key] == info[key], f"{key} is NaN"
        assert abs(info[key]) != float("inf"), f"{key} is inf"


def test_checkpoint_save_load_round_trip(tmp_path):
    agent = _agent()
    agent.learn(total_timesteps=32)

    ckpt_path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)
    assert ckpt_path.exists()

    agent2 = _agent()
    agent2.load(str(ckpt_path), load_replay_buffer=False)

    sd1, sd2 = agent.policy.state_dict(), agent2.policy.state_dict()
    assert set(sd1) == set(sd2)
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"mismatch at {key}"
    assert agent2._global_step == agent._global_step
    assert agent2._global_update == agent._global_update


def test_tdmpc2_registered_and_print_config_preflight_succeeds(capsys):
    import json

    from rl_garden.training.online import registry

    registry.run_cli(["tdmpc2", "--print-config"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["selection"]["algorithm"] == "tdmpc2"
    assert payload["selection"]["training_phase"] == "online"
