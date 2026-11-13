"""Train PPO on Pendulum with an explicit CPU rollout."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from rl_garden.algorithms import PPO
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env


# docs: start train
def train(total_timesteps: int = 2_048) -> PPO:
    env = make_gymnasium_env(
        GymnasiumEnvConfig(env_id="Pendulum-v1", num_envs=1, seed=1, device="cpu")
    )
    try:
        agent = PPO(
            env, device="cpu", num_steps=32, num_minibatches=2,
            update_epochs=2, net_arch=[64, 64], gamma=0.99,
            eval_freq=0, save_final_checkpoint=False, std_log=False,
        )
        agent.learn(total_timesteps=total_timesteps)
        return agent
    finally:
        env.close()
# docs: end train


def main(argv: Sequence[str] | None = None) -> PPO:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return train(128 if args.smoke else 2_048)


if __name__ == "__main__":
    main()
