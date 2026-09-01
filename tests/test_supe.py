from __future__ import annotations

import gymnasium as gym
import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.algorithms import OPAL, SUPE, OfflineEnvSpec
from rl_garden.algorithms.supe import load_opal_vae
from rl_garden.envs.wrappers import SkillActionWrapper

_OBS_DIM = 6
_ACTION_DIM = 2
_SKILL_DIM = 3
_CHUNK_SIZE = 3  # must match SkillActionWrapper's horizon in these tests --
# OPAL's posterior_encoder shape is fixed to the pretraining chunk_size.


def _write_opal_pretrain_h5(path) -> None:
    with h5py.File(path, "w") as f:
        for i in range(4):
            g = f.create_group(f"traj_{i}")
            g.create_dataset(
                "obs", data=np.random.randn(21, _OBS_DIM).astype(np.float32)
            )
            g.create_dataset(
                "actions",
                data=np.random.uniform(-1, 1, (20, _ACTION_DIM)).astype(np.float32),
            )
            g.create_dataset("rewards", data=np.zeros(20, dtype=np.float32))
            terminated = np.zeros(20, dtype=bool)
            terminated[-1] = True
            g.create_dataset("terminated", data=terminated)
            g.create_dataset("truncated", data=np.zeros(20, dtype=bool))


def _make_opal_checkpoint(tmp_path):
    dataset_path = tmp_path / "opal_demo.h5"
    _write_opal_pretrain_h5(dataset_path)
    env = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32),
        num_envs=1,
    )
    agent = OPAL(
        env=env, dataset_path=str(dataset_path), device="cpu", batch_size=16,
        skill_dim=_SKILL_DIM, chunk_size=_CHUNK_SIZE, hidden_size=8,
        vae_hidden_dims=(8, 8),
    )
    agent.train(3)
    ckpt_path = tmp_path / "opal.pt"
    agent.save(ckpt_path)
    return str(ckpt_path), agent


class DummyVecEnv(gym.Env):
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.broadcast_to(self.single_observation_space.low, (num_envs, _OBS_DIM)),
            high=np.broadcast_to(self.single_observation_space.high, (num_envs, _OBS_DIM)),
            dtype=np.float32,
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, _ACTION_DIM)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, _ACTION_DIM)),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        del seed
        return torch.zeros(self.num_envs, _OBS_DIM), {}

    def step(self, actions):
        obs = torch.randn(self.num_envs, _OBS_DIM)
        rewards = torch.ones(self.num_envs)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rewards, terminations, truncations, {}

    def close(self) -> None:
        return None


def _make_supe_agent(opal_checkpoint, *, horizon=_CHUNK_SIZE, num_envs=2, **overrides) -> SUPE:
    raw_env = DummyVecEnv(num_envs=num_envs)
    opal_vae = load_opal_vae(
        opal_checkpoint, _OBS_DIM, raw_env.single_action_space, device="cpu",
    )
    wrapped_env = SkillActionWrapper(
        raw_env, opal_vae.decoder, horizon=horizon, skill_dim=_SKILL_DIM, deterministic=False,
    )
    kwargs = dict(
        env=wrapped_env,
        opal_checkpoint=opal_checkpoint,
        device="cpu",
        buffer_device="cpu",
        buffer_size=64,
        batch_size=8,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
    )
    kwargs.update(overrides)
    return SUPE(**kwargs)


def test_opal_vae_matches_checkpoint_weights(tmp_path):
    opal_checkpoint, opal_agent = _make_opal_checkpoint(tmp_path)
    agent = _make_supe_agent(opal_checkpoint)
    for key, value in opal_agent.policy.state_dict().items():
        assert torch.equal(value, agent.opal_vae.state_dict()[key]), key


def test_sample_train_batch_is_inherited_not_overridden():
    assert "_sample_train_batch" not in SUPE.__dict__


def test_skill_action_space_matches_opal_skill_dim(tmp_path):
    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    agent = _make_supe_agent(opal_checkpoint)
    assert agent.env.single_action_space.shape == (_SKILL_DIM,)


def test_load_skill_relabeled_offline_buffer_aggregation(tmp_path):
    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    agent = _make_supe_agent(opal_checkpoint)

    relabel_path = tmp_path / "relabel_demo.h5"
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    terminated = np.array([False, False, True, False, False, False])
    with h5py.File(relabel_path, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("obs", data=np.random.randn(7, _OBS_DIM).astype(np.float32))
        g.create_dataset(
            "actions", data=np.random.uniform(-1, 1, (6, _ACTION_DIM)).astype(np.float32)
        )
        g.create_dataset("rewards", data=rewards)
        g.create_dataset("terminated", data=terminated)
        g.create_dataset("truncated", data=np.zeros(6, dtype=bool))

    discount = 0.9
    loaded = agent.load_skill_relabeled_offline_buffer(
        str(relabel_path), buffer_size=16, chunk_size=_CHUNK_SIZE, discount=discount,
        offline_data_ratio=0.5,
    )
    # 6-length trajectory, chunk_size=3 -> 4 valid windows (starts 0..3).
    assert loaded == 4
    assert len(agent.offline_replay_buffer) == 4

    # Hand-computed reference for window start=1: rewards=[2,3,4],
    # terminated=[F,T,F] -- termination at the window's middle position
    # zeroes the reward *after* it (position 2), matching ChunkDataset's
    # cumulative "alive" mask.
    expected_reward_1 = 2.0 * discount**0 + 3.0 * discount**1 + 4.0 * discount**2 * 0.0
    assert torch.isclose(
        agent.offline_replay_buffer.rewards[1, 0], torch.tensor(expected_reward_1), atol=1e-5
    )
    assert agent.offline_replay_buffer.dones[1, 0].item() == 1.0

    # Window start=0: rewards=[1,2,3], terminated=[F,F,T] -- termination is
    # the window's *last* step, so nothing after it to zero.
    expected_reward_0 = 1.0 * discount**0 + 2.0 * discount**1 + 3.0 * discount**2
    assert torch.isclose(
        agent.offline_replay_buffer.rewards[0, 0], torch.tensor(expected_reward_0), atol=1e-5
    )
    assert agent.offline_replay_buffer.dones[0, 0].item() == 1.0

    # Window start=3 (rewards=[4,5,6], no termination at all in-window).
    expected_reward_3 = 4.0 * discount**0 + 5.0 * discount**1 + 6.0 * discount**2
    assert torch.isclose(
        agent.offline_replay_buffer.rewards[3, 0], torch.tensor(expected_reward_3), atol=1e-5
    )
    assert agent.offline_replay_buffer.dones[3, 0].item() == 0.0


def test_truncation_does_not_zero_rewards_but_still_sets_done(tmp_path):
    # Matches ChunkDataset.create's own distinction (chunk_dataset.py:139-152,
    # d4rl_datasets.py:40,44): reward-zeroing uses termination *only*
    # (`masks`), while the aggregated `dones` output uses the broader
    # terminated-OR-truncated boundary signal.
    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    agent = _make_supe_agent(opal_checkpoint)

    relabel_path = tmp_path / "truncation_demo.h5"
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    truncated = np.array([False, True, False, False, False, False])
    with h5py.File(relabel_path, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("obs", data=np.random.randn(7, _OBS_DIM).astype(np.float32))
        g.create_dataset(
            "actions", data=np.random.uniform(-1, 1, (6, _ACTION_DIM)).astype(np.float32)
        )
        g.create_dataset("rewards", data=rewards)
        g.create_dataset("terminated", data=np.zeros(6, dtype=bool))
        g.create_dataset("truncated", data=truncated)

    discount = 0.9
    agent.load_skill_relabeled_offline_buffer(
        str(relabel_path), buffer_size=16, chunk_size=_CHUNK_SIZE, discount=discount,
        offline_data_ratio=0.5,
    )

    # Window start=1: rewards=[2,3,4], truncated (not terminated) at the
    # middle position -- reward is NOT zeroed (unlike the terminated case
    # above), but `dones` still fires for the aggregated output.
    expected_reward_1 = 2.0 * discount**0 + 3.0 * discount**1 + 4.0 * discount**2
    assert torch.isclose(
        agent.offline_replay_buffer.rewards[1, 0], torch.tensor(expected_reward_1), atol=1e-5
    )
    assert agent.offline_replay_buffer.dones[1, 0].item() == 1.0


def test_checkpoint_metadata_includes_opal_checkpoint(tmp_path):
    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    agent = _make_supe_agent(opal_checkpoint)
    assert agent._checkpoint_metadata()["opal_checkpoint"] == opal_checkpoint
