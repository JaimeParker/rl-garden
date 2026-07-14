"""HIL-SERL's ActorLoop/LearnerLoop/sync classes -- mirrors
test_real_world_serl_loops.py (base-class behavior is covered by
test_real_world_actor_loop.py / test_real_world_learner_loop.py /
test_real_world_sync.py). Unlike SERL, HilSerlActorLoop/HilSerlLearnerLoop
override behavior (intervention tagging, demo-buffer routing, crash-recovery
snapshotting), covered below."""
from __future__ import annotations

import glob
import os
import pickle

import torch
from gymnasium import spaces

from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
from rl_garden.real_world import (
    ActorSyncClient,
    LearnerSyncServer,
)
from rl_garden.real_world.actor_loop import ActorLoop
from rl_garden.real_world.hil_serl import (
    HilSerlActorLoop,
    HilSerlActorSyncClient,
    HilSerlLearnerLoop,
    HilSerlLearnerSyncServer,
)
from rl_garden.real_world.learner_loop import LearnerLoop


class _DummyVecEnv:
    num_envs = 1

    def __init__(self) -> None:
        import numpy as np

        self.single_observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.single_action_space.low[None],
            high=self.single_action_space.high[None],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        del seed
        return torch.zeros(1, 4), {}

    def step(self, actions):
        return (
            torch.randn(1, 4),
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def close(self) -> None:
        return None


def _agent() -> RLPDHybrid:
    return RLPDHybrid(
        env=_DummyVecEnv(),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
    )


def _transition(intervened: bool = False) -> dict:
    return {
        "obs": torch.zeros(1, 4),
        "next_obs": torch.zeros(1, 4),
        "action": torch.zeros(1, 3),
        "reward": torch.ones(1),
        "done": torch.zeros(1, dtype=torch.bool),
        "intervened": intervened,
    }


def test_hil_serl_loop_and_sync_classes_subclass_the_algorithm_agnostic_bases():
    assert issubclass(HilSerlActorLoop, ActorLoop)
    assert issubclass(HilSerlLearnerLoop, LearnerLoop)
    assert issubclass(HilSerlActorSyncClient, ActorSyncClient)
    assert issubclass(HilSerlLearnerSyncServer, LearnerSyncServer)


def test_hil_serl_sync_client_constructs_like_the_base():
    client = HilSerlActorSyncClient("http://127.0.0.1:6000")
    assert client._base_url == "http://127.0.0.1:6000"


def test_hil_serl_actor_loop_tags_intervened_from_info():
    loop = HilSerlActorLoop.__new__(HilSerlActorLoop)
    assert loop._extra_transition_fields({"intervene_action": torch.zeros(3)}) == {
        "intervened": True
    }
    assert loop._extra_transition_fields({}) == {"intervened": False}


def test_hil_serl_learner_loop_routes_by_intervened_flag(tmp_path):
    """Matches HIL-SERL's own actor loop (train_rlpd.py:178-193): every
    transition -- intervened or not -- goes into the online replay buffer;
    intervened transitions are *additionally* copied into the demo buffer.
    The demo buffer is a duplicated subset of the online buffer, not a
    disjoint partition of it."""
    agent = _agent()
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    loop = HilSerlLearnerLoop(agent, "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=1000)

    loop._on_transition(_transition(intervened=False))
    loop._on_transition(_transition(intervened=True))
    loop._on_transition(_transition(intervened=True))

    assert len(agent.replay_buffer) == 3
    assert len(agent.offline_replay_buffer) == 2
    assert loop.received_transitions == 3


def test_hil_serl_learner_loop_snapshots_and_reloads_on_restart(tmp_path):
    agent = _agent()
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    loop = HilSerlLearnerLoop(agent, "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2)

    loop._on_transition(_transition(intervened=False))
    loop._on_transition(_transition(intervened=True))  # crosses buffer_period=2 -> snapshot

    assert glob.glob(os.path.join(str(tmp_path), "buffer", "*.pkl"))
    assert glob.glob(os.path.join(str(tmp_path), "demo_buffer", "*.pkl"))

    reloaded_agent = _agent()
    reloaded_agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    reloaded_loop = HilSerlLearnerLoop(
        reloaded_agent, "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2
    )

    # "buffer" snapshot holds both transitions (unconditional); "demo_buffer"
    # holds the one intervened transition, duplicated -- not a remainder.
    assert len(reloaded_agent.replay_buffer) == 2
    assert len(reloaded_agent.offline_replay_buffer) == 1
    # Received-count is off the "buffer" reload only, so the intervened
    # transition isn't counted twice.
    assert reloaded_loop.received_transitions == 2


def test_hil_serl_learner_loop_preloads_demo_dataset_paths(tmp_path):
    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    demo_transitions = [_transition() for _ in range(3)]
    for t in demo_transitions:
        t.pop("intervened")
    with open(demo_dir / "session1.pkl", "wb") as f:
        pickle.dump(demo_transitions, f)

    agent = _agent()
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    loop = HilSerlLearnerLoop(
        agent,
        "127.0.0.1",
        0,
        checkpoint_dir=str(tmp_path / "ckpt"),
        buffer_period=1000,
        demo_dataset_paths=[str(demo_dir / "*.pkl")],
    )

    assert len(agent.offline_replay_buffer) == 3
    # Preloaded demo transitions don't count toward received_transitions --
    # that gates learning_starts, and HIL-SERL's own analogous gate never
    # counts --demo_path preload either, only genuine online transitions.
    assert loop.received_transitions == 0

    # Growing via live intervention keeps adding to the same buffer/slot.
    loop._on_transition(_transition(intervened=True))
    assert len(agent.offline_replay_buffer) == 4
    assert loop.received_transitions == 1


def test_hil_serl_learner_loop_reloads_demo_dataset_paths_on_every_restart(tmp_path):
    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    demo_transitions = [_transition() for _ in range(2)]
    for t in demo_transitions:
        t.pop("intervened")
    with open(demo_dir / "session1.pkl", "wb") as f:
        pickle.dump(demo_transitions, f)

    checkpoint_dir = str(tmp_path / "ckpt")
    agent = _agent()
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    HilSerlLearnerLoop(
        agent,
        "127.0.0.1",
        0,
        checkpoint_dir=checkpoint_dir,
        buffer_period=1000,
        demo_dataset_paths=[str(demo_dir / "*.pkl")],
    )
    assert len(agent.offline_replay_buffer) == 2

    # Restart: mirrors HIL-SERL's own always-reload-demo_path behavior.
    reloaded_agent = _agent()
    reloaded_agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    HilSerlLearnerLoop(
        reloaded_agent,
        "127.0.0.1",
        0,
        checkpoint_dir=checkpoint_dir,
        buffer_period=1000,
        demo_dataset_paths=[str(demo_dir / "*.pkl")],
    )
    assert len(reloaded_agent.offline_replay_buffer) == 2
