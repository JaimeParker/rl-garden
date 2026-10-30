"""State-only residual SAC on frozen ACT; the SAC update is inherited unchanged."""
from rl_garden.algorithms.residual import ResidualSAC
from rl_garden.encoders.selected_state import SelectedStateExtractor
from rl_garden.envs.robotwin.state_sources import SOURCE_FIELDS

V2_KEYS = ("state", "state_open_laptop_robot_v2")
V3_KEYS = ("state", "state_open_laptop_robot_v3")
V4_KEYS = ("state", "state_open_laptop_robot_v4")
V5_KEYS = ("state", "state_open_laptop_robot_v5")


class TaskStateResidualSACV2(ResidualSAC):
    _compatible_checkpoint_algorithms = ("TaskStateResidualSACV2",)
    state_keys = V2_KEYS
    state_version = 2
    extra_dim = 100

    def _build_features_extractor(self):
        if set(self.env.single_observation_space.spaces) != set(self.state_keys):
            raise ValueError(f"Task state v{self.state_version} requires exactly the two state keys; no RGB replay")
        if [self.env.single_observation_space[k].shape for k in self.state_keys] != [(14,), (self.extra_dim,)]:
            raise ValueError(f"Task state v{self.state_version} requires 14 + {self.extra_dim} dimensions")
        return SelectedStateExtractor(self.env.single_observation_space, keys=self.state_keys)

    def _checkpoint_metadata(self):
        meta = super()._checkpoint_metadata()
        meta["selected_state_schema"] = {
            "version": self.state_version, "actor_keys": list(self.state_keys), "critic_keys": list(self.state_keys),
            "extra_fields": list(SOURCE_FIELDS[f"open_laptop_robot_v{self.state_version}"]),
            "normalization": "native_units_no_running_statistics",
            "original_state_14_unchanged": True, "full_physics_state": False,
        }
        return meta


class TaskStateResidualSACV5(TaskStateResidualSACV2):
    _compatible_checkpoint_algorithms = ("TaskStateResidualSACV5",)
    state_keys = V5_KEYS
    state_version = 5
    extra_dim = 132

    def _checkpoint_metadata(self):
        from rl_garden.envs.robotwin.task_state_v4 import CONTACT_MAX_SEPARATION_M, CONTACT_MIN_IMPULSE_NS
        meta = super()._checkpoint_metadata()
        meta["selected_state_schema"]["contact_semantics"] = {
            "windows": ["previous_control_action_any_tick", "last_physics_tick_of_action"],
            "reset_and_zero_tick": "current_snapshot",
            "bodies": "finger_joint_descendants_only",
            "max_separation_m": CONTACT_MAX_SEPARATION_M,
            "min_impulse_Ns": CONTACT_MIN_IMPULSE_NS,
        }
        return meta


class TaskStateResidualSACV3(TaskStateResidualSACV2):
    _compatible_checkpoint_algorithms = ("TaskStateResidualSACV3",)
    state_keys = V3_KEYS
    state_version = 3
    extra_dim = 106


class TaskStateResidualSACV4(TaskStateResidualSACV2):
    _compatible_checkpoint_algorithms = ("TaskStateResidualSACV4",)
    state_keys = V4_KEYS
    state_version = 4
    extra_dim = 128

    def _checkpoint_metadata(self):
        from rl_garden.envs.robotwin.task_state_v4 import CONTACT_MAX_SEPARATION_M, CONTACT_MIN_IMPULSE_NS
        meta = super()._checkpoint_metadata()
        meta["selected_state_schema"]["contact_semantics"] = {
            "window": "previous_control_action_any_tick; reset=current_snapshot",
            "bodies": "finger_joint_descendants_only",
            "max_separation_m": CONTACT_MAX_SEPARATION_M,
            "min_impulse_Ns": CONTACT_MIN_IMPULSE_NS,
        }
        return meta
