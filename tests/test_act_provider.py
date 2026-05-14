from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.models.act.config import ACTCheckpointSpec
from rl_garden.models.act.provider import ACTBaseActionProvider


def test_visual_act_space_accepts_per_camera_rgb_keys() -> None:
    obs_space = spaces.Dict(
        {
            "rgb_base_camera": spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            ),
            "rgb_hand_camera": spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            ),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    ACTBaseActionProvider._validate_spaces(
        ACTCheckpointSpec(state_dim=5, action_dim=2, visual=True),
        obs_space,
        act_space,
    )


def test_visual_act_space_still_accepts_channel_stacked_rgb_key() -> None:
    obs_space = spaces.Dict(
        {
            "rgb": spaces.Box(low=0, high=255, shape=(64, 64, 6), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
        }
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    ACTBaseActionProvider._validate_spaces(
        ACTCheckpointSpec(state_dim=5, action_dim=2, visual=True),
        obs_space,
        act_space,
    )


def test_visual_act_stacks_per_camera_rgb_keys() -> None:
    provider = object.__new__(ACTBaseActionProvider)
    provider.device = torch.device("cpu")
    obs = {
        "rgb_hand_camera": torch.ones(2, 4, 5, 3, dtype=torch.uint8),
        "rgb_base_camera": torch.zeros(2, 4, 5, 3, dtype=torch.uint8),
    }

    rgb = provider._camera_group_to_bnc_hw(
        obs,
        legacy_key="rgb",
        prefix="rgb_",
        channels_per_camera=3,
    )

    assert rgb.shape == (2, 2, 3, 4, 5)
    # Keys are sorted for deterministic camera order: base_camera, hand_camera.
    assert torch.all(rgb[:, 0] == 0)
    assert torch.all(rgb[:, 1] == 1)
