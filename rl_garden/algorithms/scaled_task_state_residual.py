"""Residual SAC with fixed physical-unit state normalization."""
from .task_state_residual import TaskStateResidualSACV5
from rl_garden.encoders.scaled_task_state import ScaledTaskStateExtractor, PROFILE, scaling_spec


class ScaledTaskStateResidualSACV5(TaskStateResidualSACV5):
    _compatible_checkpoint_algorithms = ("ScaledTaskStateResidualSACV5",)

    def _build_features_extractor(self):
        # Retain exact key/shape validation without consuming random numbers.
        super()._build_features_extractor()
        return ScaledTaskStateExtractor(self.env.single_observation_space)

    def _checkpoint_metadata(self):
        meta = super()._checkpoint_metadata()
        fields, divisors, units = scaling_spec()
        meta["selected_state_schema"]["normalization"] = {
            "profile": PROFILE, "transform": "x/divisor", "fields": fields,
            "divisors": divisors, "units": units, "clipping": False,
            "running_statistics": False, "raw_replay_and_act_unchanged": True,
        }
        return meta
