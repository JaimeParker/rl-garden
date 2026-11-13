"""Train a small SAC agent on Pendulum using explicit CPU devices."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from rl_garden.algorithms import SAC
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env


# docs: start train
def train(total_timesteps: int = 2_000) -> SAC:
    env = make_gymnasium_env(
        GymnasiumEnvConfig(env_id="Pendulum-v1", num_envs=1, seed=1, device="cpu")
    )
    try:
        agent = SAC(
            env, device="cpu", buffer_device="cpu", buffer_size=10_000,
            learning_starts=32, batch_size=32, training_freq=4, utd=0.25,
            net_arch=[64, 64], gamma=0.99, eval_freq=0,
            save_final_checkpoint=False, std_log=False,
        )
        agent.learn(total_timesteps=total_timesteps)
        return agent
    finally:
        env.close()
# docs: end train


def main(argv: Sequence[str] | None = None) -> SAC:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return train(128 if args.smoke else 2_000)


if __name__ == "__main__":
    main()
