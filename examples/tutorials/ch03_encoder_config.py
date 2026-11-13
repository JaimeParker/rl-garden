"""Exercise an RGB/state encoder with one CPU SAC update and no renderer."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from rl_garden.algorithms import SAC
from rl_garden.encoders import EncoderConfig
from rl_garden.envs.vector_env import TorchVectorEnvAdapter


# docs: start environment
class RGBStateEnv(gym.Env):
    """Synthetic observations for checking encoder wiring, not learning quality."""

    observation_space = gym.spaces.Dict({
        "state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        "rgb_camera": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
    })
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation(), {}

    def _observation(self):
        return {
            "state": self.np_random.uniform(-1, 1, (3,)).astype(np.float32),
            "rgb_camera": self.np_random.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        }

    def step(self, action):
        return self._observation(), -float(np.square(action).sum()), False, False, {}
# docs: end environment


# docs: start train
def train() -> SAC:
    vector_env = gym.vector.SyncVectorEnv(
        [RGBStateEnv], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = TorchVectorEnvAdapter(vector_env, device="cpu")
    try:
        encoder = EncoderConfig(
            backbone="plain_conv", features_dim=16, proprio_latent_dim=8,
            plain_conv_pooling="gap",
        )
        agent = SAC(
            env, encoder_config=encoder, device="cpu", buffer_device="cpu",
            buffer_size=16, learning_starts=8, batch_size=2, training_freq=4,
            utd=0.25, net_arch=[16, 16], eval_freq=0,
            save_final_checkpoint=False, std_log=False,
        )
        agent.learn(total_timesteps=8)
        return agent
    finally:
        env.close()
# docs: end train


def main(argv: Sequence[str] | None = None) -> SAC:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the same one-update check.")
    parser.parse_args(argv)
    return train()


if __name__ == "__main__":
    main()
