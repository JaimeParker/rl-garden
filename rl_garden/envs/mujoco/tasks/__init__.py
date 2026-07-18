"""rl-garden's custom MuJoCo tasks. Importing this package triggers each
task module's ``gym.register()`` call. Imported (not at rl_garden.envs
package scope) from inside make_mujoco_env's per-env-instance thunk -- see
rl_garden.envs.mujoco.env module docstring."""
from rl_garden.envs.mujoco.tasks import inverted_pendulum_custom  # noqa: F401

__all__: list[str] = []
