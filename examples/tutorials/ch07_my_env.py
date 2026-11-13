"""Register a one-dimensional Gymnasium task and train SAC on CPU."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from rl_garden.algorithms import SAC
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env


# docs: start environment
class LineReachEnv(gym.Env):
    """Move a point toward zero; the time limit is supplied at registration."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.position = np.zeros(1, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.position = self.np_random.uniform(-1, 1, (1,)).astype(np.float32)
        return self.position.copy(), {}

    def step(self, action: np.ndarray):
        movement = 0.1 * np.clip(action, -1, 1)
        self.position = np.clip(self.position + movement, -1, 1).astype(np.float32)
        distance = float(abs(self.position[0]))
        return self.position.copy(), -distance, distance < 0.02, False, {"distance": distance}
# docs: end environment


# docs: start registration
ENV_ID = "TutorialLineReach-v0"
if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=LineReachEnv, max_episode_steps=32)
# docs: end registration


# docs: start train
def train(total_timesteps: int) -> SAC:
    env = make_gymnasium_env(
        GymnasiumEnvConfig(env_id=ENV_ID, num_envs=1, seed=1, device="cpu")
    )
    try:
        agent = SAC(
            env, device="cpu", buffer_device="cpu", buffer_size=512,
            batch_size=16, learning_starts=16, training_freq=4, utd=0.25,
            net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
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
    return train(64 if args.smoke else 256)


if __name__ == "__main__":
    main()
