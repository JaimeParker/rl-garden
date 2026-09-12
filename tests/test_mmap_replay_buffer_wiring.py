"""SAC/RLPD/RLPD_hybrid wiring for `mmap_dir`/`mmap_mode` disk-backed replay
buffers -- the buffer classes themselves (`ReplayBuffer`,
`NStepReplayBuffer`) already support this and are covered by
tests/test_replay_buffer.py; these tests only check that the algorithm
constructors actually pass `mmap_dir`/`mmap_mode` through, and that the
guard/override behavior ported from DDPG (tests/test_drqv2_components.py)
holds for SAC too.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from rl_garden.algorithms.rlpd import RLPD
from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
from rl_garden.algorithms.sac import SAC
from rl_garden.buffers.replay_buffer import ReplayBuffer
from rl_garden.buffers.memory_efficient_buffer import MemoryEfficientReplayBuffer
from rl_garden.buffers.nstep_buffer import NStepReplayBuffer


class DummyDictVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


class DummyBoxVecEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def _sac_kwargs(**overrides):
    kwargs = dict(
        env=DummyDictVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        net_arch={"pi": [16], "qf": [16]},
    )
    kwargs.update(overrides)
    return kwargs


def test_sac_builds_mmap_buffer(tmp_path):
    agent = SAC(**_sac_kwargs(mmap_dir=tmp_path))
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None
    assert (tmp_path / "manifest.json").is_file()


def test_sac_builds_mmap_nstep_buffer(tmp_path):
    agent = SAC(**_sac_kwargs(mmap_dir=tmp_path, nstep=2))
    assert isinstance(agent.replay_buffer, NStepReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None


def test_sac_rejects_mmap_replay_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="cannot be embedded"):
        SAC(**_sac_kwargs(mmap_dir=tmp_path, save_replay_buffer=True))


def test_sac_builds_mmap_buffer_from_box_env(tmp_path):
    # DummyBoxVecEnv's bare Box observation space is boundary-normalized to
    # Dict by BaseAlgorithm.__init__ (see rl_garden.envs.wrappers
    # .VectorizedDictStateWrapper) before SAC ever sees it, so mmap_dir now
    # works from a Box-origin env instead of raising.
    agent = SAC(
        env=DummyBoxVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=16,
        batch_size=4,
        learning_starts=0,
        training_freq=1,
        eval_freq=0,
        net_arch=[16],
        mmap_dir=tmp_path,
    )
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None


def test_sac_load_rejects_mmap_replay_checkpoint(tmp_path):
    agent = SAC(**_sac_kwargs(mmap_dir=tmp_path))
    with pytest.raises(ValueError, match="not supported with mmap buffers"):
        agent.load(tmp_path / "nonexistent.pt", load_replay_buffer=True)


def test_sac_load_replay_buffer_rejects_mmap(tmp_path):
    agent = SAC(**_sac_kwargs(mmap_dir=tmp_path))
    with pytest.raises(ValueError, match="not supported with mmap buffers"):
        agent.load_replay_buffer(tmp_path / "nonexistent.pt")


def test_rlpd_builds_mmap_buffer(tmp_path):
    agent = RLPD(**_sac_kwargs(mmap_dir=tmp_path))
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None


def test_rlpd_hybrid_builds_mmap_buffer(tmp_path):
    agent = RLPDHybrid(**_sac_kwargs(mmap_dir=tmp_path, discrete_hidden_dim=8))
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None


def test_rlpd_hybrid_grasp_penalty_buffer_supports_mmap(tmp_path):
    agent = RLPDHybrid(
        **_sac_kwargs(mmap_dir=tmp_path, discrete_hidden_dim=8, use_grasp_penalty=True)
    )
    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None
    assert agent.replay_buffer.store_grasp_penalty is True


class DummyStackedDictVecEnv:
    """Dict obs with a frame-stacked image key, as MemoryEfficientReplayBuffer
    requires (leading dim == frame_stack)."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.single_observation_space = spaces.Dict(
            {
                "rgb_cam": spaces.Box(low=0, high=255, shape=(2, 32, 32, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )


def test_rlpd_hybrid_memory_efficient_buffer_supports_mmap(tmp_path):
    agent = RLPDHybrid(
        **_sac_kwargs(
            env=DummyStackedDictVecEnv(),
            mmap_dir=tmp_path,
            discrete_hidden_dim=8,
            memory_efficient_buffer=True,
            memory_efficient_frame_stack=2,
        )
    )
    assert isinstance(agent.replay_buffer, MemoryEfficientReplayBuffer)
    assert agent.replay_buffer._mmap_store is not None
    assert (tmp_path / "manifest.json").is_file()
