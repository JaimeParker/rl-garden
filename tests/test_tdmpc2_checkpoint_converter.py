"""Tests for tools/conversion/convert_tdmpc2_checkpoint.py on a synthetic
upstream-style TD-MPC2 multitask state_dict (incl. keys that must be dropped)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "conversion"))

from convert_tdmpc2_checkpoint import convert_state_dict, infer_architecture, main  # noqa: E402
from test_pwm_smoke import ACTION_DIMS, OBS_DIMS, TASKS, WM_ARCH, make_world_model  # noqa: E402

from rl_garden.algorithms.pwm import load_pwm_world_model  # noqa: E402

_DROPPED = [
    "_Qs.params.0.weight",
    "_action_masks",
    "_detach_Qs_params.0.weight",
    "_pi.0.weight",
    "_target_Qs_params.0.weight",
    "_termination.0.weight",
    "log_std_dif",
    "log_std_min",
]


def _upstream_state_dict(with_action_masks: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(upstream_style, rl_garden_reference)``."""
    reference = make_world_model(seed=3).state_dict()
    upstream: dict[str, torch.Tensor] = {}
    for key, value in reference.items():
        if key.startswith("encoder."):
            upstream["_encoder.state." + key[len("encoder."):]] = value
        elif key == "task_emb.weight":
            upstream["_task_emb.weight"] = value
        elif key == "_action_masks":
            if with_action_masks:
                upstream[key] = value
        else:
            upstream[key] = value
    upstream.update(
        {
            "_pi.0.weight": torch.zeros(4, 4),
            "_Qs.params.0.weight": torch.zeros(4, 4),
            "_detach_Qs_params.0.weight": torch.zeros(4, 4),
            "_target_Qs_params.0.weight": torch.zeros(4, 4),
            "_termination.0.weight": torch.zeros(1, 4),
            "log_std_min": torch.tensor(-10.0),
            "log_std_dif": torch.tensor(12.0),
        }
    )
    return upstream, reference


def test_converts_and_strict_loads():
    upstream, reference = _upstream_state_dict()
    converted, manifest, dropped = convert_state_dict(upstream, tasks=TASKS)

    assert dropped == sorted(_DROPPED)
    assert set(converted) == set(reference)
    for key, value in reference.items():
        torch.testing.assert_close(converted[key], value)
    assert manifest["action_dims"] == ACTION_DIMS
    assert manifest["obs_dim"] == max(OBS_DIMS)
    assert manifest["action_dim"] == max(ACTION_DIMS)
    for key in ("task_dim", "latent_dim", "enc_dim", "num_enc_layers", "mlp_dim", "num_bins"):
        assert manifest[key] == WM_ARCH[key], key
    assert manifest["num_tasks"] == len(TASKS)


def test_infers_deeper_encoder():
    from rl_garden.world_models.multitask_latent_consistency import MultitaskWorldModel

    model = MultitaskWorldModel(3, OBS_DIMS, ACTION_DIMS, task_dim=8, latent_dim=16, enc_dim=8, num_enc_layers=4, mlp_dim=16)
    assert infer_architecture(model.state_dict())["num_enc_layers"] == 4


def test_requires_action_dims_without_action_masks():
    upstream, _ = _upstream_state_dict(with_action_masks=False)
    with pytest.raises(ValueError, match="action dims are unknown"):
        convert_state_dict(upstream)
    converted, manifest, _ = convert_state_dict(upstream, action_dims=ACTION_DIMS)
    assert manifest["action_dims"] == ACTION_DIMS


def test_rejects_inconsistent_action_dims():
    upstream, _ = _upstream_state_dict()
    with pytest.raises(ValueError, match="inconsistent"):
        convert_state_dict(upstream, action_dims=[2, 2, 2])


def test_cli_writes_output_pwm_can_load(tmp_path, monkeypatch, capsys):
    upstream, reference = _upstream_state_dict()
    source = tmp_path / "agent.pt"
    torch.save({"model": upstream}, source)
    dst = tmp_path / "converted"
    monkeypatch.setattr(sys, "argv", ["convert", "--src", str(source), "--dst_dir", str(dst)])
    main()

    assert "dropped 8 keys" in capsys.readouterr().out
    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["simnorm_dim"] == 8 and manifest["vmin"] == -10.0 and manifest["vmax"] == 10.0
    world_model, source_info = load_pwm_world_model(dst)
    assert source_info["format"] == "converted_tdmpc2_checkpoint"
    for key, value in reference.items():
        torch.testing.assert_close(world_model.state_dict()[key], value)
