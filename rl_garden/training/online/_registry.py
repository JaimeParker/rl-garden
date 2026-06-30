from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class OnlineAlgorithmRegistry(BaseAlgorithmRegistry):
    package_name = "rl_garden.training.online"
    phase_name = "online"


registry = OnlineAlgorithmRegistry()
