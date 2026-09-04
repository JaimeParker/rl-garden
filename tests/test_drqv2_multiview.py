import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.encoders.drqv2_conv import DrQv2Encoder
from rl_garden.encoders.drqv2_multiview import (
    IndependentLateFusionDrQv2Encoder,
    MultiViewStemDrQv2Encoder,
    SpatialLateFusionDrQv2Encoder,
)


def _space(channels: int = 9) -> spaces.Box:
    return spaces.Box(0, 255, shape=(channels, 64, 64), dtype=np.uint8)


@pytest.mark.parametrize(
    "encoder_class",
    [
        MultiViewStemDrQv2Encoder,
        SpatialLateFusionDrQv2Encoder,
        IndependentLateFusionDrQv2Encoder,
    ],
)
def test_multiview_encoder_shape_backward_and_finite(encoder_class) -> None:
    encoder = encoder_class(_space())
    obs = torch.randint(0, 256, (2, 9, 64, 64), dtype=torch.uint8)
    output = encoder(obs)

    assert encoder.features_dim == 20_000
    assert output.shape == (2, 20_000)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    gradients = [parameter.grad for parameter in encoder.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_multiview_encoder_parameter_counts_stay_close_to_b9b() -> None:
    baseline = DrQv2Encoder(_space())
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())

    for encoder in (
        MultiViewStemDrQv2Encoder(_space()),
        SpatialLateFusionDrQv2Encoder(_space()),
        IndependentLateFusionDrQv2Encoder(_space()),
    ):
        count = sum(parameter.numel() for parameter in encoder.parameters())
        assert 0.9 * baseline_count <= count <= 1.1 * baseline_count


def test_spatial_late_fusion_starts_view_permutation_invariant() -> None:
    torch.manual_seed(7)
    encoder = SpatialLateFusionDrQv2Encoder(_space())
    obs = torch.randint(0, 256, (2, 9, 64, 64), dtype=torch.uint8)
    permuted = torch.cat((obs[:, 6:9], obs[:, 0:3], obs[:, 3:6]), dim=1)

    torch.testing.assert_close(encoder(obs), encoder(permuted))


def test_multiview_encoders_reject_non_three_view_input() -> None:
    with pytest.raises(ValueError, match="exactly three RGB views"):
        MultiViewStemDrQv2Encoder(_space(channels=6))
    with pytest.raises(ValueError, match="exactly three RGB views"):
        SpatialLateFusionDrQv2Encoder(_space(channels=12))
    with pytest.raises(ValueError, match="exactly three RGB views"):
        IndependentLateFusionDrQv2Encoder(_space(channels=6))


def test_independent_late_fusion_does_not_share_camera_parameters() -> None:
    encoder = IndependentLateFusionDrQv2Encoder(_space())
    first_weights = [convnet[0].weight for convnet in encoder.view_convnets]

    assert len({weight.data_ptr() for weight in first_weights}) == 3
    assert all(weight.shape == first_weights[0].shape for weight in first_weights)
