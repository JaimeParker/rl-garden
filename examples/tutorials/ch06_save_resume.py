"""Save SAC with replay, reload it, compare actions, and continue on CPU."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rl_garden.algorithms import SAC
from rl_garden.common.checkpoint import load_checkpoint_file
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env


def make_agent(env) -> SAC:
    return SAC(
        env, device="cpu", buffer_device="cpu", buffer_size=256,
        learning_starts=16, batch_size=8, training_freq=2, utd=0.5,
        net_arch=[32, 32], eval_freq=0, std_log=False,
        save_final_checkpoint=False,
    )


# docs: start roundtrip
def save_resume(directory: Path, steps: int = 50) -> dict[str, int]:
    env = make_gymnasium_env(
        GymnasiumEnvConfig(env_id="Pendulum-v1", num_envs=1, seed=1, device="cpu")
    )
    try:
        original = make_agent(env)
        original.learn(total_timesteps=steps)
        observation, _ = env.reset(seed=7)
        original.policy.eval()
        with torch.no_grad():
            expected = original.policy.predict(observation, deterministic=True)

        checkpoint = original.save(directory / "agent.pt", include_replay_buffer=True)
        metadata = load_checkpoint_file(checkpoint, map_location="cpu")["metadata"]
        assert metadata["algorithm_class"] == "SAC"
        assert metadata["global_step"] == steps
        print("Algorithm:", metadata["algorithm_class"])
        print("Actor factory:", metadata["hyperparameters"].get("actor_factory", "in-tree"))
        restored = make_agent(env)
        restored.load(checkpoint, load_replay_buffer=True)
        restored.policy.eval()
        with torch.no_grad():
            actual = restored.policy.predict(observation, deterministic=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert restored._global_step == original._global_step
        assert restored.replay_buffer.size == original.replay_buffer.size

        updates_before = restored._global_update
        restored.learn(total_timesteps=steps + 16)
        assert restored._global_update > updates_before
        return {"saved_step": original._global_step, "resumed_step": restored._global_step}
    finally:
        env.close()
# docs: end roundtrip


def main(argv: Sequence[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    steps = 50 if args.smoke else 200
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return save_resume(args.output_dir, steps)
    with TemporaryDirectory(prefix="rl-garden-checkpoint-") as directory:
        return save_resume(Path(directory), steps)


if __name__ == "__main__":
    main()
