# tests/test_combined_extractor_augmentation_cache.py
from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.combined import CombinedExtractor


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def _batch() -> dict:
    return {
        "rgb": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "state": torch.randn(2, 4),
    }


def test_two_extractors_prepare_batch_do_not_clobber_each_other():
    space = _obs_space()
    actor_ext = CombinedExtractor(
        space, image_keys=("rgb",), image_augmentation="random_shift",
        random_shift_pad=2, augmentation_seed=1,
    )
    critic_ext = CombinedExtractor(
        space, image_keys=("rgb",), image_augmentation="random_shift",
        random_shift_pad=2, augmentation_seed=2,
    )
    obs = _batch()
    actor_ext.prepare_batch(obs)
    critic_ext.prepare_batch(obs)

    # Both cache entries must survive in the same obs dict, under distinct
    # keys -- neither prepare_batch() call may overwrite the other's.
    actor_cached = obs[actor_ext._aug_stack_key]
    critic_cached = obs[critic_ext._aug_stack_key]
    assert actor_ext._aug_stack_key != critic_ext._aug_stack_key
    # Different augmentation seeds over random pixel input -> different crops.
    assert not torch.equal(actor_cached, critic_cached)

    # Each extractor reads back its own cached tensor unchanged.
    out_actor = actor_ext.extract(obs)
    out_critic = critic_ext.extract(obs)
    assert out_actor.shape == out_critic.shape
