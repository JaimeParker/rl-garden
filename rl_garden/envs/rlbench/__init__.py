"""RLBench env backend package.

Unlike ``mujoco``/``ogbench`` (delegate straight to ``gymnasium.make``),
this package builds ``rlbench.environment.Environment``/``TaskEnvironment``
itself, so it can flatten/rename RLBench's native observation into
rl-garden's conventions -- see ``rl_garden.buffers.rlbench_dataset``'s
module docstring for why, and ``docs/guides/rlbench-integration.md`` for
install steps (CoppeliaSim + PyRep, no pip extra).
"""
from rl_garden.envs.rlbench.config import RLBenchEnvConfig
from rl_garden.envs.rlbench.env import RLBenchGymEnv, make_rlbench_env

__all__ = ["RLBenchEnvConfig", "RLBenchGymEnv", "make_rlbench_env"]
