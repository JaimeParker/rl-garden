"""Executable CPU example for tutorial 08."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

# docs: start component
from rl_garden.networks import DeterministicTanhActor


def make_deterministic_actor(features_dim, action_space):
    return DeterministicTanhActor(features_dim, action_space, hidden_dims=[32, 32])
# docs: end component


# docs: start train
def train(steps: int):
    from rl_garden.algorithms import SAC
    from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env
    from rl_garden.policies.contracts import validate_sac_actor

    env = make_gymnasium_env(GymnasiumEnvConfig("Pendulum-v1", 1, 1, device="cpu"))
    try:
        capabilities = validate_sac_actor(make_deterministic_actor(3, env.single_action_space), 3, env.single_action_space)
        assert not capabilities.stochastic and not capabilities.log_prob_is_density
        agent = SAC(
            env, actor_factory=make_deterministic_actor, ent_coef=0.0,
            target_policy_noise=0.2, target_noise_clip=0.5, actor_q_agg="first",
            device="cpu", buffer_device="cpu", buffer_size=256, batch_size=8,
            learning_starts=8, training_freq=4, utd=0.25, net_arch=[32, 32],
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
