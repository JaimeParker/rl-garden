"""Random-shift augmentation of ``(T, B)`` observation windows (RecurrentSAC/TransformerSAC):
one shift per sequence, the same at every step and for the sequence's final observations."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import RecurrentSAC, TransformerSAC
from rl_garden.common.obs_utils import flatten_leading_dims, index_obs
from rl_garden.encoders import CombinedExtractor, RandomShiftsAug
from rl_garden.encoders.config import EncoderConfig
from rl_garden.observations import ObservationSchema

STEPS, BATCH, SIZE = 5, 6, 16


def _space(cameras=("rgb_cam",), frames=None):
    image = (frames, SIZE, SIZE, 3) if frames else (SIZE, SIZE, 3)
    return spaces.Dict(
        {
            **{camera: spaces.Box(0, 255, image, dtype=np.uint8) for camera in cameras},
            "state": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        }
    )


def _extractor(space, fusion_mode="stack_channels"):
    config = EncoderConfig(
        backbone="plain_conv",
        features_dim=8,
        plain_conv_pooling="gap",
        image_fusion_mode=fusion_mode,
        image_augmentation="random_shift",
    )
    return CombinedExtractor(space, ObservationSchema.from_space(space), config, augmentation_seed=0)


def _augmented(extractor, obs):
    if extractor.fusion_mode == "stack_channels":
        return {"stacked": obs[extractor._aug_stack_key]}
    return obs[extractor._aug_images_key]


def _random_obs(space, leading):
    return {
        key: torch.randint(0, 256, leading + sub.shape, dtype=torch.uint8)
        if sub.dtype == np.uint8
        else torch.randn(leading + sub.shape)
        for key, sub in space.spaces.items()
    }


@pytest.mark.parametrize(
    "cameras,frames,fusion_mode",
    [
        (("rgb_cam",), None, "stack_channels"),
        (("rgb_cam",), 2, "stack_channels"),
        (("rgb_a", "rgb_b"), None, "per_key"),
    ],
)
def test_each_sequence_and_its_final_obs_get_one_shift(cameras, frames, fusion_mode):
    space = _space(cameras, frames)
    extractor = _extractor(space, fusion_mode)
    per_sequence = _random_obs(space, (1, BATCH))
    window = {key: value.expand(STEPS, *value.shape[1:]).clone() for key, value in per_sequence.items()}
    final_columns = torch.tensor([4, 0, 4, 2])
    final_obs = {key: value[0, final_columns] for key, value in per_sequence.items()}

    extractor.prepare_batch(window)
    extractor.prepare_window_final_obs(window, final_obs, final_columns)

    for name, images in _augmented(extractor, window).items():
        assert images.shape[:2] == (STEPS, BATCH)
        assert torch.equal(images, images[:1].expand_as(images))
        assert not torch.equal(images[0, 0], images[0, 1:].mean(0))
        assert torch.equal(_augmented(extractor, final_obs)[name], images[0, final_columns])


def test_window_features_read_the_augmented_images():
    space = _space()
    extractor = _extractor(space)
    window = _random_obs(space, (STEPS, BATCH))
    plain = flatten_leading_dims(window)
    extractor.prepare_batch(window)
    assert not torch.allclose(extractor.extract(flatten_leading_dims(window)), extractor.extract(plain))
    sliced = index_obs(window, slice(0, 2))
    assert torch.allclose(
        extractor.extract(flatten_leading_dims(sliced)),
        extractor.extract(flatten_leading_dims(window))[: 2 * BATCH],
        atol=1e-5,
    )


@pytest.mark.parametrize("fusion_mode", ["stack_channels", "per_key"])
def test_flat_batches_keep_their_independent_shifts(fusion_mode):
    space = _space(("rgb_a", "rgb_b"))
    extractor = _extractor(space, fusion_mode)
    obs, next_obs = _random_obs(space, (BATCH,)), _random_obs(space, (BATCH,))
    extractor.prepare_batch(obs, next_obs)

    generator = torch.Generator().manual_seed(0)
    shift = RandomShiftsAug(extractor.random_shift_pad)

    def expected(batch):
        images = {key: batch[key].float().div(255.0).permute(0, 3, 1, 2) for key in ("rgb_a", "rgb_b")}
        if fusion_mode == "stack_channels":
            images = {"stacked": torch.cat(list(images.values()), dim=1)}
        return {name: shift(image, generator=generator) for name, image in images.items()}

    for batch in (obs, next_obs):
        for name, image in expected(batch).items():
            assert torch.equal(_augmented(extractor, batch)[name], image)


class _PixelEnv:
    """RGB + state observations; env ``i`` truncates every ``3 + i`` steps."""

    num_envs = 2

    def __init__(self):
        self.single_observation_space = _space()
        self.observation_space = self.single_observation_space
        self.single_action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2, 2), np.float32)
        self.lengths = torch.tensor([3, 4])
        self.steps = torch.zeros(2, dtype=torch.long)

    def _obs(self):
        return _random_obs(self.single_observation_space, (2,))

    def reset(self, seed=None, options=None):
        self.steps.zero_()
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        ended = self.steps >= self.lengths
        final, obs = self._obs(), self._obs()
        self.steps[ended] = 0
        return obs, torch.ones(2), torch.zeros_like(ended), ended, {"final_observation": final}


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RecurrentSAC, {"rnn_hidden_size": 8}),
        (TransformerSAC, {"embed_dim": 8, "head_dim": 4, "num_heads": 2, "num_transformer_layers": 1, "memory_len": 2}),
    ],
)
def test_sequence_sac_trains_on_augmented_pixel_windows(cls, kwargs, monkeypatch):
    finals = []
    prepare = CombinedExtractor.prepare_window_final_obs

    def record(self, obs, final_obs, final_columns):
        prepare(self, obs, final_obs, final_columns)
        finals.append(self._aug_stack_key in final_obs)

    monkeypatch.setattr(CombinedExtractor, "prepare_window_final_obs", record)
    torch.manual_seed(0)
    agent = cls(
        _PixelEnv(),
        encoder_config=EncoderConfig(
            backbone="plain_conv", features_dim=8, plain_conv_pooling="gap", image_augmentation="random_shift"
        ),
        burn_in_len=2, learning_len=2, forward_len=1, buffer_size=2 * 40, buffer_device="cpu", device="cpu",
        learning_starts=16, training_freq=2, batch_size=4, net_arch=[16], eval_freq=0, std_log=False,
        save_final_checkpoint=False, **kwargs,
    )
    agent.learn(2 * 24)
    assert agent._global_update > 0
    assert finals and all(finals)
