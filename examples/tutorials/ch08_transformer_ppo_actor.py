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


class TransformerGaussianActor(nn.Module):
    """Splits the features into tokens, runs a transformer over them and reads a
    diagonal Gaussian over the action from the pooled output."""

    def __init__(self, features_dim: int, action_space: gym.spaces.Box,
                 d_model: int = 16, num_tokens: int = 2, num_layers: int = 1) -> None:
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.num_tokens, self.d_model = num_tokens, d_model
        self.to_tokens = nn.Linear(features_dim, num_tokens * d_model)
        # dropout=0: PPO compares the log-prob of the rollout with the update's,
        # which a stochastic forward would bias.
        layer = nn.TransformerEncoderLayer(d_model, nhead=2, dim_feedforward=32, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.mean = nn.Linear(d_model, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.register_buffer("low", torch.as_tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(action_space.high, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> Normal:
        tokens = self.to_tokens(features).view(-1, self.num_tokens, self.d_model)
        mean = self.mean(self.transformer(tokens).mean(dim=1))
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @staticmethod
    def _reduce(log_prob, entropy, sum_dims):
        if sum_dims:
            return log_prob.sum(-1, keepdim=True), entropy.sum(-1, keepdim=True)
        return log_prob, entropy

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        distribution = self(features)
        action = distribution.mean if deterministic else distribution.sample()
        log_prob, entropy = self._reduce(distribution.log_prob(action), distribution.entropy(), sum_dims)
        return action, log_prob, entropy

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        distribution = self(features)
        return self._reduce(distribution.log_prob(actions), distribution.entropy(), sum_dims)

    def deterministic_action(self, features):
        return self(features).mean

    def clamp_action(self, actions):
        return actions.clamp(self.low, self.high)


def make_transformer_actor(features_dim: int, action_space: gym.spaces.Box) -> nn.Module:
    return TransformerGaussianActor(features_dim, action_space)
# docs: end component


# docs: start train
def train(steps: int):
    from rl_garden.algorithms import PPO
    from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env
    from rl_garden.policies.contracts import validate_ppo_actor

    env = make_gymnasium_env(GymnasiumEnvConfig("Pendulum-v1", 1, 1, device="cpu"))
    try:
        actor = make_transformer_actor(3, env.single_action_space)
        capabilities = validate_ppo_actor(actor, 3, env.single_action_space)
        assert capabilities.is_gaussian and capabilities.supports_per_dim_log_prob
        agent = PPO(
            env, actor_factory=make_transformer_actor, device="cpu", num_steps=8,
            num_minibatches=2, update_epochs=1, net_arch=[32, 32],
            eval_freq=0, save_final_checkpoint=False, std_log=False,
        )
        agent.learn(total_timesteps=steps)
        return agent
    finally:
        env.close()
# docs: end train


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return train(16 if args.smoke else 200)


if __name__ == "__main__":
    main()
