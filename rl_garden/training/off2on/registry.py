from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry


class Off2OnAlgorithmRegistry(BaseAlgorithmRegistry):
    package_name = "rl_garden.training.off2on.entrypoints"
    phase_name = "off2on"


registry = Off2OnAlgorithmRegistry()
