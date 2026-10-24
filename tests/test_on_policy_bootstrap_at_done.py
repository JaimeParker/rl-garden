"""On-policy GAE bootstrap at episode ends follows ``bootstrap_at_done``.

``OnPolicyAlgorithm.learn`` stores ``V(final_observation)`` in
``RolloutBuffer.final_values`` only for the envs ``_bootstrap_mask`` selects:
truncations by default, every done with ``"always"``, none with ``"never"``.
"""

from __future__ import annotations

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import PPO, RecurrentPPO

# Env 0 terminates, env 1 truncates, env 2 does both, env 3 keeps running --
# all on the first step.
_TERMINATIONS = torch.tensor([True, False, True, False])
_TRUNCATIONS = torch.tensor([False, True, True, False])
_FINAL_STATE = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10.0


class _EpisodeEndVecEnv(gym.Env):
    num_envs = 4
    single_observation_space = spaces.Dict(
        {"state": spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)}
    )
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        self.observation_space = self.single_observation_space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_envs, 2), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        del seed, options
        return {"state": torch.zeros(self.num_envs, 4)}, {}

    def step(self, action):
        del action
        obs = {"state": -torch.ones(self.num_envs, 4)}
        rewards = torch.zeros(self.num_envs)
        infos = {"final_observation": {"state": _FINAL_STATE.clone()}}
        return obs, rewards, _TERMINATIONS.clone(), _TRUNCATIONS.clone(), infos

    def close(self) -> None:
        return None


def _ppo_kwargs(**overrides) -> dict:
    kwargs = {
        "device": "cpu",
        "num_steps": 1,
        "num_minibatches": 1,
        "update_epochs": 1,
        "eval_freq": 0,
        "log_freq": 0,
        "std_log": False,
        "target_kl": None,
        "net_arch": [8],
    }
    kwargs.update(overrides)
    return kwargs


def _skip_update() -> dict[str, float]:
    return {}


def _state_sum_values(obs, *args, **kwargs):
    del args, kwargs
    return obs["state"].sum(dim=-1, keepdim=True)


@pytest.mark.parametrize(
    "bootstrap_at_done,bootstrapped_envs",
    [
        ("truncated", [False, True, False, False]),
        ("always", [True, True, True, False]),
        ("never", [False, False, False, False]),
    ],
)
def test_ppo_rollout_final_values_follow_bootstrap_at_done(
    bootstrap_at_done, bootstrapped_envs
):
    agent = PPO(_EpisodeEndVecEnv(), **_ppo_kwargs(bootstrap_at_done=bootstrap_at_done))
    agent.policy.predict_values = _state_sum_values
    agent.train = _skip_update

    agent.learn(total_timesteps=agent.num_envs)

    final_state_values = _FINAL_STATE.sum(dim=-1)
    mask = torch.tensor(bootstrapped_envs)
    expected = torch.where(mask, final_state_values, torch.zeros(4))
    assert torch.allclose(agent.rollout_buffer.final_values[0], expected)
    # The episode cut itself is independent of bootstrap_at_done.
    assert agent.rollout_buffer.dones[0].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_bootstrap_at_done_defaults_to_truncated_and_rejects_unknown():
    agent = PPO(_EpisodeEndVecEnv(), **_ppo_kwargs())
    assert agent.bootstrap_at_done == "truncated"
    assert agent._checkpoint_metadata()["bootstrap_at_done"] == "truncated"
    with pytest.raises(AssertionError):
        PPO(_EpisodeEndVecEnv(), **_ppo_kwargs(bootstrap_at_done="sometimes"))


def test_recurrent_final_values_with_truncation_subset_mask():
    """SequencePPO's override indexes the recurrent state with the bootstrap
    mask; a strict subset of the done envs must give each selected env the
    same value as a full-batch evaluation."""
    agent = RecurrentPPO(_EpisodeEndVecEnv(), **_ppo_kwargs(), rnn_hidden_size=8)
    obs, _ = agent.env.reset(seed=0)
    hidden = agent._initial_hidden_state(agent.num_envs)
    actions, _, _, _, hidden = agent._rollout_step(
        obs, hidden, torch.ones(agent.num_envs)
    )
    _, _, terminations, truncations, infos = agent.env.step(agent._env_action(actions))
    mask = agent._bootstrap_mask(terminations, truncations)
    assert mask.tolist() == [False, True, False, False]

    final_values = agent._compute_final_values(infos, mask, hidden)

    full_values, _ = agent._predict_values_recurrent(
        agent._obs_to_policy_device(infos["final_observation"]),
        hidden,
        torch.zeros(agent.num_envs),
    )
    expected = torch.where(mask, full_values.view(-1), torch.zeros(agent.num_envs))
    assert torch.allclose(final_values, expected, atol=1e-6)


@pytest.mark.parametrize(
    "algorithm,module,builder",
    [
        ("ppo", "rl_garden.training.online.ppo", "build_ppo"),
        ("recurrent_ppo", "rl_garden.training.online.recurrent_ppo", "build_recurrent_ppo"),
        (
            "transformer_ppo",
            "rl_garden.training.online.transformer_ppo",
            "build_transformer_ppo",
        ),
        ("gail", "rl_garden.training.online.gail", "build_gail"),
        ("dppo", "rl_garden.training.online.dppo", "build_dppo"),
        ("flow_ppo", "rl_garden.training.online.flow_ppo", "build_flow_ppo"),
        (
            "diffusion_cm_distill_online",
            "rl_garden.training.online.diffusion_cm_distill",
            "build_diffusion_cm_distill",
        ),
    ],
)
def test_bootstrap_at_done_cli_reaches_constructor(
    monkeypatch, algorithm, module, builder
):
    from rl_garden.training import inspection
    from rl_garden.training.online import registry

    args = registry.parse_args(
        [algorithm, "--log-type", "none", "--bootstrap-at-done", "never"]
    )
    assert args.bootstrap_at_done == "never"
    assert registry.parse_args([algorithm, "--log-type", "none"]).bootstrap_at_done == (
        "truncated"
    )

    captured: dict = {}

    def _capture_constructor_kwargs(target, /, **kwargs):
        captured.update(kwargs)
        return target

    monkeypatch.setattr(inspection, "construct_agent", _capture_constructor_kwargs)
    getattr(importlib.import_module(module), builder)(
        args, _EpisodeEndVecEnv(), None, None, None
    )
    assert captured["bootstrap_at_done"] == "never"
