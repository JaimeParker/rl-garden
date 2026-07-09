from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class RealWorldAlgorithmRegistry(BaseAlgorithmRegistry):
    package_name = "rl_garden.training.real_world"
    phase_name = "real_world"


registry = RealWorldAlgorithmRegistry()
