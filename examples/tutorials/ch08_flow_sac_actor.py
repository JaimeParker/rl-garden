"""Executable CPU example for tutorial 08."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

# docs: start component
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from rl_garden.networks import SACActorBase


class EulerFlowActor(SACActorBase):
    log_prob_is_density = False

    def __init__(self, features_dim: int, action_space: gym.spaces.Box,
                 steps: int = 2, noise_std: float = 0.3, hidden: int = 32) -> None:
        super().__init__()
        self.act_dim = int(np.prod(action_space.shape))
        self.steps, self.noise_std = steps, noise_std
        self.velocity = nn.Sequential(
            nn.Linear(features_dim + self.act_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.act_dim),
        )
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("middle", (high + low) / 2)
        self.register_buffer("half_range", (high - low) / 2)

    def _integrate(self, features, noisy: bool):
        batch = features.shape[0]
        x = torch.randn(batch, self.act_dim, device=features.device) if noisy \
            else features.new_zeros(batch, self.act_dim)
        log_prob = Normal(0.0, 1.0).log_prob(x).sum(-1, keepdim=True)
        for k in range(self.steps):
            t = features.new_full((batch, 1), k / self.steps)
            mean = x + self.velocity(torch.cat([features, x, t], dim=-1)) / self.steps
            if noisy:
                x = mean + self.noise_std * torch.randn_like(mean)  # reparameterized
                log_prob = log_prob + Normal(mean, self.noise_std).log_prob(x).sum(-1, keepdim=True)
            else:
                x = mean
        return self.middle + self.half_range * torch.tanh(x), log_prob

    def action_log_prob(self, features):
        return self._integrate(features, noisy=True)

    def deterministic_action(self, features):
        return self._integrate(features, noisy=False)[0]


def make_flow_actor(features_dim: int, action_space: gym.spaces.Box) -> nn.Module:
    return EulerFlowActor(features_dim, action_space)
# docs: end component


# docs: start train
def train(steps: int):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from ch04_offline_bc import write_dataset
    from rl_garden.algorithms import SAC, CalQL, OfflineEnvSpec
    from rl_garden.buffers.h5_dataset import infer_specs_from_h5, load_h5_dataset_to_replay_buffer
    from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env
    from rl_garden.policies.contracts import validate_sac_actor

    env = make_gymnasium_env(GymnasiumEnvConfig("Pendulum-v1", 1, 1, device="cpu"))
    try:
        capabilities = validate_sac_actor(make_flow_actor(3, env.single_action_space), 3, env.single_action_space)
        assert capabilities.stochastic and not capabilities.log_prob_is_density
        agent = SAC(
            env, actor_factory=make_flow_actor, device="cpu", buffer_device="cpu",
            buffer_size=256, batch_size=8, learning_starts=8, training_freq=4,
            utd=0.25, net_arch=[32, 32], eval_freq=0,
            save_final_checkpoint=False, std_log=False,
        )
        agent.learn(total_timesteps=steps)
    finally:
        env.close()
    with TemporaryDirectory(prefix="rlgarden-flow-") as directory:
        path = write_dataset(Path(directory))
        observation_space, action_space = infer_specs_from_h5(path, action_low=-2, action_high=2)
        offline = CalQL(
            OfflineEnvSpec(observation_space, action_space), actor_factory=make_flow_actor,
            cql_importance_sample=False, device="cpu", buffer_device="cpu",
            buffer_size=256, batch_size=8, net_arch=[32, 32], n_critics=2,
            critic_subsample_size=2, use_compile=False, eval_freq=0,
            save_final_checkpoint=False, std_log=False,
        )
        load_h5_dataset_to_replay_buffer(offline.replay_buffer, path)
        offline.learn(total_timesteps=1)
    return agent, offline
# docs: end train


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return train(16 if args.smoke else 200)


if __name__ == "__main__":
    main()
