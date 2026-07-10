from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from gymnasium import spaces

from rl_garden.models.reward.success.model import SuccessClassifier, load_classifier_fn

_OBS_SPACE = spaces.Dict(
    {
        "wrist": spaces.Box(0, 255, (3, 128, 128), dtype=np.uint8),
        "state": spaces.Box(-1, 1, (20,), dtype=np.float32),
    }
)


def _make_checkpoint() -> str:
    model = SuccessClassifier(_OBS_SPACE, image_keys=["wrist"], pretrained_weights=None)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(model.state_dict(), path)
    return path


def test_forward_returns_raw_logit_not_probability():
    model = SuccessClassifier(_OBS_SPACE, image_keys=["wrist"], pretrained_weights=None)
    obs = {"wrist": torch.randint(0, 255, (2, 3, 128, 128), dtype=torch.uint8).float()}
    logit = model(obs)
    assert logit.shape == (2,)
    # The head's final layer is a plain nn.Linear with no activation --
    # confirms forward() is a raw logit, not already sigmoid-applied.
    assert isinstance(model.head[-1], torch.nn.Linear)


def test_load_classifier_fn_applies_sigmoid_and_extracts_image_keys():
    path = _make_checkpoint()
    try:
        fn = load_classifier_fn(path, _OBS_SPACE, image_keys=["wrist"])
        obs = {
            "wrist": torch.randint(0, 255, (3, 3, 128, 128), dtype=torch.uint8).float(),
            "state": torch.zeros(3, 20),  # must be ignored -- classifier is image-only
        }
        prob = fn(obs)
        assert prob.shape == (3,)
        assert torch.all((prob >= 0) & (prob <= 1))
    finally:
        os.unlink(path)


def test_load_classifier_fn_does_not_require_pretrained_weights_file():
    # pretrained_weights=None inside load_classifier_fn: the checkpoint's
    # state_dict already has every parameter, so loading must not touch
    # rl_garden/encoders/resnet.py's pretrained-weights file lookup at all.
    path = _make_checkpoint()
    try:
        load_classifier_fn(path, _OBS_SPACE, image_keys=["wrist"])
    finally:
        os.unlink(path)
