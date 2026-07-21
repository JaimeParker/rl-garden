# tests/test_dual_encoder.py
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from rl_garden.encoders.dual_encoder import build_secondary_extractor
from rl_garden.encoders import CombinedExtractor


def _dict_space() -> spaces.Dict:
    # PlainConv's default "flatten" pooling only supports specific fixed
    # input sizes (its 4-layer max-pool stack assumes 64x64 or 128x128) --
    # an 8x8 image dies partway through pooling regardless of this task's
    # change (same constraint Task 2 hit). Use 64x64 so tests that route
    # through the real PlainConv factory are actually exercisable.
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def test_returns_none_when_nothing_configured():
    result = build_secondary_extractor(
        full_observation_space=_dict_space(),
        features_extractor_class=CombinedExtractor,
        primary_kwargs={"image_keys": ("rgb",), "state_key": "state"},
        extra_obs_keys=(),
        override_image_encoder_factory=None,
    )
    assert result is None


def test_builds_extractor_when_extra_obs_keys_set():
    result = build_secondary_extractor(
        full_observation_space=_dict_space(),
        features_extractor_class=CombinedExtractor,
        primary_kwargs={"image_keys": ("rgb",), "state_key": "state"},
        extra_obs_keys=("state",),
        override_image_encoder_factory=None,
    )
    assert isinstance(result, CombinedExtractor)


def test_builds_extractor_with_overridden_image_encoder_factory():
    from rl_garden.encoders import BaseFeaturesExtractor
    import torch

    class _Marker(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=11):
            super().__init__(observation_space, features_dim)

        def forward(self, obs):
            return torch.zeros(obs.shape[0], self.features_dim)

    def factory(img_space):
        return _Marker(img_space)

    result = build_secondary_extractor(
        full_observation_space=_dict_space(),
        features_extractor_class=CombinedExtractor,
        primary_kwargs={"image_keys": ("rgb",), "state_key": "state"},
        extra_obs_keys=(),
        override_image_encoder_factory=factory,
    )
    assert isinstance(result, CombinedExtractor)
    assert isinstance(result.image_encoder, _Marker)


def test_primary_kwargs_image_encoder_factory_preserved_when_not_overridden():
    def factory(img_space):
        from rl_garden.encoders import PlainConv

        # img_space is channels-first (C, H, W) here (CombinedExtractor
        # hands the factory a stacked CHW Box) -- image_size wants (H, W),
        # i.e. shape[1:], not shape[:2].
        return PlainConv(img_space, features_dim=13, image_size=img_space.shape[1:])

    result = build_secondary_extractor(
        full_observation_space=_dict_space(),
        features_extractor_class=CombinedExtractor,
        primary_kwargs={
            "image_keys": ("rgb",),
            "state_key": "state",
            "image_encoder_factory": factory,
        },
        extra_obs_keys=("state",),
        override_image_encoder_factory=None,
    )
    assert result.image_encoder.features_dim == 13
