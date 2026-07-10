"""rl_garden/models/reward/ was previously unimported dead code with two
latent import-time bugs (a broken import path in classifiers/hsv, plus
sklearn/torchvision imported eagerly at module top instead of lazily) --
regression coverage so it stays importable."""
from __future__ import annotations


def test_models_reward_package_imports_cleanly():
    import rl_garden.models.reward as reward

    assert reward.SuccessClassifier is not None


def test_classifiers_hsv_submodule_imports_cleanly():
    from rl_garden.models.reward.classifiers import hsv

    assert hsv.CompressedHDF5LabelGenerator is not None
