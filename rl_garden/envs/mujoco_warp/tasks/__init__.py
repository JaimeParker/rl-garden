"""rl-garden's custom mujoco_warp tasks. Importing this package triggers each
task module's register_mujoco_warp_task() call."""
from rl_garden.envs.mujoco_warp.tasks import inverted_pendulum_warp  # noqa: F401

__all__: list[str] = []
