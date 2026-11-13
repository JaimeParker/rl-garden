"""Executable CPU example for tutorial 08."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

# docs: start component
import torch
from gymnasium import spaces
from torch import nn

from rl_garden.encoders.base import BaseFeaturesExtractor


class BackboneExtractor(BaseFeaturesExtractor):
    """Images and proprioception -> one (B, features_dim) vector."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 16) -> None:
        super().__init__(observation_space, features_dim)
        self.image_keys = [k for k in observation_space.spaces if k.startswith("rgb_")]
        self.state_keys = [k for k in observation_space.spaces if k.startswith("state")]
        # Stand-in for a pretrained backbone (load your checkpoint here and
        # freeze it with requires_grad_(False) if it should stay fixed).
        self.vision = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.vision.requires_grad_(False)
        state_dim = sum(observation_space.spaces[k].shape[-1] for k in self.state_keys)
        self.project = nn.Linear(16 * len(self.image_keys) + state_dim, features_dim)

    def forward(self, obs) -> torch.Tensor:
        parts = [
            self.vision(obs[k].permute(0, 3, 1, 2).float() / 255.0)  # (B, H, W, C) uint8
            for k in self.image_keys
        ]
        parts += [obs[k].float() for k in self.state_keys]
        return torch.relu(self.project(torch.cat(parts, dim=-1)))
# docs: end component


# docs: start train
def train():
    import gymnasium as gym
    from ch03_encoder_config import RGBStateEnv
    from rl_garden.algorithms import SAC
    from rl_garden.envs.vector_env import TorchVectorEnvAdapter
    from rl_garden.policies.contracts import validate_sac_actor

    vector_env = gym.vector.SyncVectorEnv(
        [RGBStateEnv], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env = TorchVectorEnvAdapter(vector_env, device="cpu")
    try:
        agent = SAC(
            env, policy_kwargs={"actor_extractor_class": BackboneExtractor},
            encoder_sharing="shared_critic_grad", device="cpu", buffer_device="cpu",
            buffer_size=16, learning_starts=8, batch_size=2, training_freq=4,
            utd=0.25, net_arch=[16, 16], eval_freq=0,
            save_final_checkpoint=False, std_log=False,
        )
        validate_sac_actor(agent.policy.actor, 16, env.single_action_space)
        agent.learn(total_timesteps=8)
        return agent
    finally:
        env.close()
# docs: end train


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the same one-update check.")
    parser.parse_args(argv)
    return train()


if __name__ == "__main__":
    main()
