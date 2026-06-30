from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class OfflineAlgorithmRegistry(BaseAlgorithmRegistry):
    package_name = "rl_garden.training.offline"
    phase_name = "offline"


registry = OfflineAlgorithmRegistry()
