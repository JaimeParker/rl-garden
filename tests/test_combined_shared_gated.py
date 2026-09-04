import unittest

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.base import BaseFeaturesExtractor
from rl_garden.encoders.combined import CombinedExtractor


class _ChannelMeanEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box) -> None:
        super().__init__(observation_space, features_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def _factory(space: spaces.Box) -> BaseFeaturesExtractor:
    return _ChannelMeanEncoder(space)


def _observation_space() -> spaces.Dict:
    entries = {
        key: spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
        for key in ("rgb", "rgb_left_wrist", "rgb_right_wrist")
    }
    entries["state"] = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
    return spaces.Dict(entries)


class SharedGatedFusionTest(unittest.TestCase):
    def test_one_encoder_and_initial_mean_fusion(self) -> None:
        extractor = CombinedExtractor(
            _observation_space(),
            image_keys=("rgb", "rgb_left_wrist", "rgb_right_wrist"),
            image_encoder_factory=_factory,
            fusion_mode="shared_gated",
        )
        self.assertIsNotNone(extractor.image_encoder)
        self.assertEqual(len(extractor.image_encoders), 0)
        self.assertEqual(extractor.features_dim, 67)
        self.assertIsNotNone(extractor.view_embeddings)
        with torch.no_grad():
            extractor.view_embeddings.zero_()

        obs = {
            "rgb": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
            "rgb_left_wrist": torch.full((2, 8, 8, 3), 127, dtype=torch.uint8),
            "rgb_right_wrist": torch.full((2, 8, 8, 3), 255, dtype=torch.uint8),
            "state": torch.zeros(2, 14),
        }
        out = extractor(obs)
        expected = torch.full((2, 3), (0.0 + 127.0 / 255.0 + 1.0) / 3.0)
        self.assertEqual(tuple(out.shape), (2, 67))
        torch.testing.assert_close(out[:, :3], expected)

    def test_rejects_mismatched_camera_shapes(self) -> None:
        observation_space = _observation_space()
        observation_space.spaces["rgb_right_wrist"] = spaces.Box(
            0, 255, shape=(10, 8, 3), dtype=np.uint8
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            CombinedExtractor(
                observation_space,
                image_keys=("rgb", "rgb_left_wrist", "rgb_right_wrist"),
                image_encoder_factory=_factory,
                fusion_mode="shared_gated",
            )


if __name__ == "__main__":
    unittest.main()
