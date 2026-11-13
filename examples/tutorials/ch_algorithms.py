"""Construct one small CPU agent per algorithm family, without training."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from rl_garden.algorithms import BC, IQL, PPO, SAC, OfflineEnvSpec, WSRL
from rl_garden.algorithms.tdmpc2 import TDMPC2
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env


# docs: start ppo
def build_ppo(env) -> PPO:
    return PPO(
        env, device="cpu", num_steps=8, num_minibatches=2, update_epochs=1,
        net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end ppo


# docs: start sac
def build_sac(env) -> SAC:
    return SAC(
        env, device="cpu", buffer_device="cpu", buffer_size=32, batch_size=4,
        net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end sac


# docs: start iql
def build_iql(spec: OfflineEnvSpec) -> IQL:
    return IQL(
        spec, device="cpu", buffer_device="cpu", buffer_size=32, batch_size=4,
        net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end iql


# docs: start wsrl
def build_wsrl(spec: OfflineEnvSpec) -> WSRL:
    return WSRL(
        spec, device="cpu", buffer_device="cpu", buffer_size=32, batch_size=4,
        net_arch=[32, 32], n_critics=2, critic_subsample_size=2,
        use_compile=False, eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end wsrl


# docs: start bc
def build_bc(spec: OfflineEnvSpec) -> BC:
    return BC(
        spec, device="cpu", buffer_device="cpu", buffer_size=32, batch_size=4,
        net_arch=[32, 32], eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end bc


# docs: start tdmpc2
def build_tdmpc2(env) -> TDMPC2:
    return TDMPC2(
        env, device="cpu", buffer_device="cpu", buffer_size=32, batch_size=2,
        episode_length=200, horizon=2, latent_dim=32, mlp_dim=32, simnorm_dim=8,
        num_q=2, num_bins=11, num_samples=8, num_elites=2, num_pi_trajs=2,
        iterations=1, eval_freq=0, save_final_checkpoint=False, std_log=False,
    )
# docs: end tdmpc2


# docs: start setup
def construct_agents() -> dict:
    env = make_gymnasium_env(GymnasiumEnvConfig(
        env_id="Pendulum-v1", num_envs=1, seed=1, device="cpu",
    ))
    try:
        spec = OfflineEnvSpec(env.single_observation_space, env.single_action_space)
        return {
            "ppo": build_ppo(env), "sac": build_sac(env),
            "iql": build_iql(spec), "wsrl": build_wsrl(spec), "bc": build_bc(spec),
            "tdmpc2": build_tdmpc2(env),
        }
    finally:
        env.close()
# docs: end setup


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use the same construction-only budget.")
    parser.parse_args(argv)
    agents = construct_agents()
    print("Constructed on CPU without training: " + ", ".join(agents))
    return agents


if __name__ == "__main__":
    main()
