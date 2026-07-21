from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class HITLAlgorithmRegistry(BaseAlgorithmRegistry):
    package_name = "rl_garden.training.hitl"
    phase_name = "hitl"


registry = HITLAlgorithmRegistry()
