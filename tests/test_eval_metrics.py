"""``masked_episode_metrics`` reads completed-episode stats through ``_final_info``."""
from __future__ import annotations

import pytest
import torch

from rl_garden.common.eval_metrics import masked_episode_metrics


def test_no_final_info_returns_empty():
    assert masked_episode_metrics({"log": {"x": 1.0}}) == {}


def test_final_info_without_episode_returns_empty():
    infos = {
        "final_info": {"elapsed_steps": torch.tensor([3, 4])},
        "_final_info": torch.tensor([True, False]),
    }

    assert masked_episode_metrics(infos) == {}


def test_final_info_without_mask_raises():
    infos = {"final_info": {"episode": {"return": torch.tensor([1.0, 2.0])}}}

    with pytest.raises(ValueError, match="_final_info"):
        masked_episode_metrics(infos)


def test_returns_done_rows_under_canonical_keys():
    infos = {
        "final_info": {
            "episode": {
                "r": torch.tensor([10.0, 99.0, 30.0]),
                "_r": torch.tensor([True, True, True]),
                "t": torch.tensor([0.1, 0.2, 0.3]),
                "success_once": torch.tensor([True, False, False]),
            }
        },
        "_final_info": torch.tensor([True, False, True]),
    }

    metrics = masked_episode_metrics(infos)

    assert set(metrics) == {"return", "success_once"}
    torch.testing.assert_close(metrics["return"], torch.tensor([10.0, 30.0]))
    torch.testing.assert_close(metrics["success_once"], torch.tensor([True, False]))


def test_no_done_rows_returns_empty():
    infos = {
        "final_info": {"episode": {"return": torch.tensor([1.0, 2.0])}},
        "_final_info": torch.tensor([False, False]),
    }

    assert masked_episode_metrics(infos) == {}
