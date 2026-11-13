"""Write synthetic Pendulum-shaped transitions and exercise offline BC on CPU."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from rl_garden.algorithms import BC, OfflineEnvSpec
from rl_garden.buffers.h5_dataset import infer_specs_from_h5, load_h5_dataset_to_replay_buffer


# docs: start dataset
def write_dataset(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pendulum.h5"
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as dataset:
        dataset.attrs["purpose"] = "synthetic format fixture, not expert demonstrations"
        for episode in range(2):
            length = 64
            theta = rng.uniform(-np.pi, np.pi, length + 1)
            velocity = rng.uniform(-1, 1, length + 1)
            observations = np.column_stack((np.cos(theta), np.sin(theta), velocity))
            actions = rng.uniform(-1, 1, (length, 1)).astype(np.float32)
            rewards = -(theta[:-1] ** 2 + 0.1 * velocity[:-1] ** 2)
            truncated = np.zeros(length, dtype=bool)
            truncated[-1] = True
            trajectory = dataset.create_group(f"traj_{episode}")
            trajectory.create_dataset("obs", data=observations.astype(np.float32))
            trajectory.create_dataset("actions", data=actions)
            trajectory.create_dataset("rewards", data=rewards.astype(np.float32))
            trajectory.create_dataset("terminated", data=np.zeros(length, dtype=bool))
            trajectory.create_dataset("truncated", data=truncated)
    return path
# docs: end dataset


# docs: start train
def train(path: Path, updates: int) -> BC:
    observation_space, action_space = infer_specs_from_h5(path, action_low=-2, action_high=2)
    agent = BC(
        OfflineEnvSpec(observation_space, action_space), device="cpu",
        buffer_device="cpu", buffer_size=256, batch_size=16,
        net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
    loaded = load_h5_dataset_to_replay_buffer(agent.replay_buffer, path)
    assert loaded == 128
    agent.learn(total_timesteps=updates)
    return agent
# docs: end train


def main(argv: Sequence[str] | None = None) -> BC:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    path = write_dataset(args.output_dir)
    agent = train(path, 4 if args.smoke else 20)
    print(f"dataset={path} transitions=128 updates={agent._global_update}")
    return agent


if __name__ == "__main__":
    main()
