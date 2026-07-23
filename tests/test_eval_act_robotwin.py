from __future__ import annotations

import sys

import pytest
import torch

from examples.eval_act_robotwin import _env_request, _split_action_summary, parse_args


def test_robotwin_control_mode_defaults_to_legacy_joint_pos(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_act_robotwin.py"])

    args = parse_args()

    assert args.control_mode == "joint_pos"
    assert _env_request(args).control_mode == "joint_pos"


@pytest.mark.parametrize("mode", ["joint_pos", "ee_pose"])
def test_robotwin_control_mode_is_explicitly_selectable(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_act_robotwin.py", "--robotwin-control-mode", mode],
    )

    args = parse_args()

    assert args.control_mode == mode
    assert _env_request(args).control_mode == mode


def test_legacy_control_mode_flag_remains_compatible(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_act_robotwin.py", "--control-mode", "ee_pose"],
    )

    assert parse_args().control_mode == "ee_pose"


def test_robotwin_control_mode_rejects_residual_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_act_robotwin.py", "--robotwin-control-mode", "ee_delta_pose"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_robotwin_control_mode_rejects_conflicting_legacy_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_act_robotwin.py",
            "--robotwin-control-mode",
            "ee_pose",
            "--control-mode",
            "joint_pos",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_split_action_summary_supports_absolute_ee_pose() -> None:
    summary = _split_action_summary(torch.arange(14), "ee_pose")

    assert summary["left_xyz_mean"] == 1.0
    assert summary["left_rotvec_abs_max"] == 5.0
    assert summary["left_gripper_mean"] == 6.0
    assert summary["right_xyz_mean"] == 8.0
    assert summary["right_rotvec_abs_max"] == 12.0
    assert summary["right_gripper_mean"] == 13.0


def test_split_action_summary_rejects_wrong_ee_pose_width() -> None:
    assert _split_action_summary(torch.zeros(16), "ee_pose") == {}
