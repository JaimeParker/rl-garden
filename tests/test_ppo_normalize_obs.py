"""Tests for online running observation normalization (PPO's normalize_obs).

Covers the new RunningObsNormalizer primitive directly, its wiring into
FlattenExtractor/CombinedExtractor, and end-to-end PPO(normalize_obs=True)
behavior (stats move during rollout, checkpoint round-trips them, image keys
are unaffected).
"""
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden.algorithms.ppo import PPO
from rl_garden.common.obs_normalization import RunningObsNormalizer
from rl_garden.encoders.combined import CombinedExtractor
from rl_garden.encoders.flatten import FlattenExtractor


class _FakeBoxEnv:
    def __init__(self, num_envs: int = 3, episode_len: int = 5, obs_dim: int = 5) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.obs_dim = obs_dim
        self._t = torch.zeros(num_envs, dtype=torch.long)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _obs(self):
        return torch.randn(self.num_envs, self.obs_dim)

    def reset(self, seed=None):
        del seed
        self._t.zero_()
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.episode_len
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        reward = torch.ones(self.num_envs)
        self._t[terminated] = 0
        return self._obs(), reward, terminated, truncated, {}


def test_running_obs_normalizer_matches_welford_reference():
    torch.manual_seed(0)
    normalizer = RunningObsNormalizer(dim=4)
    batches = [torch.randn(6, 4) for _ in range(5)]
    for batch in batches:
        normalizer.update(batch)
    all_data = torch.cat(batches, dim=0)
    ref_mean = all_data.mean(dim=0)
    ref_std = all_data.std(dim=0, unbiased=False)
    assert torch.allclose(normalizer._mean.squeeze(0), ref_mean, atol=1e-4)
    assert torch.allclose(normalizer._std.squeeze(0), ref_std, atol=1e-4)


def test_running_obs_normalizer_update_is_noop_in_eval_mode():
    normalizer = RunningObsNormalizer(dim=3)
    normalizer.eval()
    normalizer.update(torch.randn(4, 3))
    assert normalizer.count.item() == 0
    assert torch.equal(normalizer._mean, torch.zeros(1, 3))


def test_running_obs_normalizer_state_dict_round_trips():
    normalizer = RunningObsNormalizer(dim=3)
    normalizer.update(torch.randn(8, 3))
    state = normalizer.state_dict()

    fresh = RunningObsNormalizer(dim=3)
    fresh.load_state_dict(state)
    assert torch.equal(fresh._mean, normalizer._mean)
    assert torch.equal(fresh._std, normalizer._std)
    assert fresh.count.item() == normalizer.count.item()


def test_flatten_extractor_normalize_obs_off_by_default():
    obs_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
    extractor = FlattenExtractor(obs_space)
    assert extractor.normalizer is None
    extractor.update_normalizer(torch.randn(5, 4))  # no-op, must not raise


def test_flatten_extractor_normalizes_when_enabled():
    obs_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
    extractor = FlattenExtractor(obs_space, normalize_obs=True)
    extractor.train()
    for _ in range(3):
        extractor.update_normalizer(torch.randn(6, 4))
    assert extractor.normalizer.count.item() == 18
    out = extractor(torch.zeros(1, 4))
    assert not torch.equal(out, torch.zeros(1, 4))  # shifted by the running mean


def test_combined_extractor_normalizes_vector_and_state_but_not_images():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (64, 64, 3), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "extra": spaces.Box(-np.inf, np.inf, (3,), np.float32),
        }
    )
    extractor = CombinedExtractor(obs_space, image_keys=("rgb",), normalize_obs=True)
    extractor.train()
    assert "state" in extractor._obs_normalizers
    assert "extra" in extractor._obs_normalizers
    assert "rgb" not in extractor._obs_normalizers

    obs = {
        "rgb": torch.randint(0, 256, (5, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(5, 4) * 10 + 3,
        "extra": torch.randn(5, 3) * 2 - 1,
    }
    extractor.update_normalizer(obs)
    assert extractor._obs_normalizers["state"].count.item() == 5
    assert extractor._obs_normalizers["extra"].count.item() == 5
    extractor(obs)  # forward with normalization applied must not raise


def test_combined_extractor_normalize_obs_off_is_unchanged():
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (64, 64, 3), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        }
    )
    extractor = CombinedExtractor(obs_space, image_keys=("rgb",), normalize_obs=False)
    assert len(extractor._obs_normalizers) == 0
    obs = {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }
    extractor.update_normalizer(obs)  # no-op, must not raise


def test_ppo_normalize_obs_stats_move_during_rollout():
    env = _FakeBoxEnv()
    agent = PPO(
        env, num_steps=8, num_minibatches=2, update_epochs=1, device="cpu",
        normalize_obs=True, eval_freq=0, log_freq=0, net_arch=[16],
    )
    assert agent.policy.features_extractor.normalizer.count.item() == 0
    agent.learn(total_timesteps=8 * env.num_envs * 2)
    assert agent.policy.features_extractor.normalizer.count.item() > 0


def test_ppo_normalize_obs_checkpoint_round_trip(tmp_path):
    env = _FakeBoxEnv()
    agent = PPO(
        env, num_steps=8, num_minibatches=2, update_epochs=1, device="cpu",
        normalize_obs=True, eval_freq=0, log_freq=0, net_arch=[16],
    )
    agent.learn(total_timesteps=8 * env.num_envs * 2)
    mean_before = agent.policy.features_extractor.normalizer._mean.clone()
    count_before = agent.policy.features_extractor.normalizer.count.item()

    path = agent.save(tmp_path / "ckpt.pt", include_replay_buffer=False)
    resumed = PPO(
        env, num_steps=8, num_minibatches=2, update_epochs=1, device="cpu",
        normalize_obs=True, eval_freq=0, log_freq=0, net_arch=[16],
    )
    resumed.load(path, load_replay_buffer=False)
    assert torch.equal(resumed.policy.features_extractor.normalizer._mean, mean_before)
    assert resumed.policy.features_extractor.normalizer.count.item() == count_before


def test_normalize_obs_reaches_extractor_through_cli_args_entrypoint():
    """normalize_obs must survive PPOArgs -> _ppo_common_kwargs ->
    construct_agent(PPO, ...) unfiltered, not just PPO(normalize_obs=...)
    called directly (every other test in this file bypasses the CLI args
    layer)."""
    from rl_garden.training.online.ppo import PPOArgs, build_ppo

    env = _FakeBoxEnv()
    args = PPOArgs(
        lr_schedule="constant", normalize_obs=True, num_steps=8, num_minibatches=2,
        update_epochs=1, eval_freq=0, log_freq=0, obs_mode="state",
    )
    agent = build_ppo(args, env, None, None, None)
    assert agent.normalize_obs is True
    assert agent.policy.features_extractor.normalizer is not None
