from __future__ import annotations

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from rl_garden.algorithms import OPAL, SUPE, OfflineEnvSpec
from rl_garden.algorithms.supe import build_frozen_opal_vae, load_opal_vae, opal_vae_record
from rl_garden.envs.vector_env import TorchVectorEnvAdapter
from rl_garden.envs.wrappers import SkillActionWrapper
from rl_garden.observations import ObservationContractError

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
        state_space = spaces.Box(-1.0, 1.0, (_OBS_DIM,), dtype=np.float32)
        self.single_observation_space = spaces.Dict({"state": state_space})
        batched_state_space = spaces.Box(
            low=np.broadcast_to(state_space.low, (num_envs, _OBS_DIM)),
            high=np.broadcast_to(state_space.high, (num_envs, _OBS_DIM)),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({"state": batched_state_space})
        self.single_action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(self.single_action_space.low, (num_envs, _ACTION_DIM)),
            high=np.broadcast_to(self.single_action_space.high, (num_envs, _ACTION_DIM)),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        del seed
        return {"state": torch.zeros(self.num_envs, _OBS_DIM)}, {}

    def step(self, actions):
        obs = {"state": torch.randn(self.num_envs, _OBS_DIM)}
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
    kwargs = {
        "env": wrapped_env,
        "opal_checkpoint": opal_checkpoint,
        "device": "cpu",
        "buffer_device": "cpu",
        "buffer_size": 64,
        "batch_size": 8,
        "learning_starts": 1,
        "training_freq": 4,
        "eval_freq": 0,
        "log_freq": 0,
        "net_arch": [8],
    }
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


class _SingleEnv(gym.Env):
    def __init__(self, obs_keys: tuple[str, ...]) -> None:
        self.observation_space = spaces.Dict(
            {key: spaces.Box(-np.inf, np.inf, (_OBS_DIM,), np.float32) for key in obs_keys}
        )
        self.action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), np.float32)

    def _obs(self):
        return {key: np.zeros(_OBS_DIM, np.float32) for key in self.observation_space}

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 1.0, False, False, {}


def _vector_env(*obs_keys: str) -> TorchVectorEnvAdapter:
    return TorchVectorEnvAdapter(
        SyncVectorEnv(
            [lambda: _SingleEnv(obs_keys) for _ in range(2)], autoreset_mode=AutoresetMode.SAME_STEP
        ),
        device="cpu",
    )


def test_entrypoint_wraps_a_state_only_vector_env(tmp_path, monkeypatch):
    from rl_garden.training.online.entrypoints import supe

    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    monkeypatch.setattr("rl_garden.common.utils.get_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(
        "rl_garden.training.inspection.construct_agent",
        lambda cls, **kwargs: (kwargs["env"], kwargs["eval_env"]),
    )
    args = supe.SUPEArgs(opal_checkpoint=opal_checkpoint, horizon=_CHUNK_SIZE)

    for env in supe.build_supe(args, _vector_env("state"), _vector_env("state"), None, None):
        env.reset()
        assert env.single_action_space.shape == (_SKILL_DIM,)
        _, reward, *_ = env.step(torch.zeros(2, _SKILL_DIM))
        assert reward.tolist() == [float(_CHUNK_SIZE)] * 2
        env.close()


def test_entrypoint_rejects_an_env_that_is_not_state_only():
    from rl_garden.training.online.entrypoints import supe

    with pytest.raises(ObservationContractError, match="state_goal"):
        supe.build_supe(supe.SUPEArgs(), _vector_env("state", "state_goal"), None, None, None)


def test_checkpoint_embeds_the_opal_vae_and_restores_the_decoder_the_env_uses(tmp_path):
    opal_checkpoint, opal_agent = _make_opal_checkpoint(tmp_path)
    _make_supe_agent(opal_checkpoint).save(tmp_path / "supe.pt")

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_checkpoint, _ = _make_opal_checkpoint(other_dir)
    raw_env = DummyVecEnv()
    vae = load_opal_vae(other_checkpoint, _OBS_DIM, raw_env.single_action_space, device="cpu")
    env = SkillActionWrapper(
        raw_env, vae.decoder, horizon=_CHUNK_SIZE, skill_dim=_SKILL_DIM, deterministic=False
    )
    agent = _make_supe_agent(other_checkpoint, env=env, opal_vae=vae)
    agent.load(tmp_path / "supe.pt", load_replay_buffer=False)
    for key, value in opal_agent.policy.decoder.state_dict().items():
        assert torch.equal(env.decoder.state_dict()[key], value), key


def test_opal_record_rebuilds_the_vae(tmp_path):
    opal_checkpoint, _ = _make_opal_checkpoint(tmp_path)
    vae = load_opal_vae(
        opal_checkpoint, _OBS_DIM, DummyVecEnv().single_action_space, device="cpu"
    )
    record = opal_vae_record(vae)
    assert record["hyperparameters"] == {
        "skill_dim": _SKILL_DIM, "chunk_size": _CHUNK_SIZE, "hidden_size": 8,
        "vae_hidden_dims": [8, 8],
    }
    rebuilt = build_frozen_opal_vae(
        record["hyperparameters"], record["state_dict"], _OBS_DIM,
        DummyVecEnv().single_action_space, device="cpu",
    )
    assert not any(p.requires_grad for p in rebuilt.parameters())


def test_missing_opal_checkpoint_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="gone.pt"):
        load_opal_vae(
            str(tmp_path / "gone.pt"), _OBS_DIM, DummyVecEnv().single_action_space, device="cpu"
        )
