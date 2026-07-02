"""Tests for the offline Q-contrast diagnostic probe."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import h5py
import numpy as np
import torch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnostics"
    / "probe_q_contrast.py"
)
_SPEC = importlib.util.spec_from_file_location("probe_q_contrast", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


class ToyPolicy:
    def extract_features(self, obs, stop_gradient: bool = False):
        del stop_gradient
        return obs

    def predict(self, obs, deterministic: bool = False):
        del deterministic
        return torch.zeros(obs.shape[0], 2, device=obs.device)

    def q_values_all(self, features, actions, target: bool = False):
        del target
        q1 = (features[:, :2] * actions).sum(dim=-1)
        q2 = q1 + actions.pow(2).sum(dim=-1)
        return torch.stack([q1, q2])


def test_summary_stats_reports_percentiles():
    stats = probe.summary_stats(
        torch.tensor([1.0, 2.0, 3.0]),
        prefix="probe",
        percentiles=(10.0, 50.0, 90.0),
    )

    assert stats["probe/mean"] == 2.0
    assert stats["probe/min"] == 1.0
    assert stats["probe/max"] == 3.0
    assert stats["probe/p50"] == 2.0
    assert "probe/p10" in stats
    assert "probe/p90" in stats


def test_uniform_actions_respects_bounds_and_shape():
    actions = probe.uniform_actions(
        batch_size=3,
        num_actions=5,
        action_low=torch.tensor([-1.0, -0.5]),
        action_high=torch.tensor([1.0, 0.5]),
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert actions.shape == (3, 5, 2)
    assert torch.all(actions >= torch.tensor([-1.0, -0.5]))
    assert torch.all(actions <= torch.tensor([1.0, 0.5]))


def test_compute_action_grad_norm_for_min_q():
    policy = ToyPolicy()
    obs = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    actions = torch.tensor([[0.5, -0.25], [0.1, 0.2]])

    grad_norm = probe.compute_action_grad_norm(
        policy, obs, actions, ensemble_reduce="min"
    )

    assert torch.allclose(grad_norm, torch.tensor([5.0, 2.0]))


def test_compute_q_contrast_metrics_smoke():
    metrics = probe.compute_q_contrast_metrics(
        ToyPolicy(),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]),
        action_low=torch.tensor([-1.0, -1.0]),
        action_high=torch.tensor([1.0, 1.0]),
        config=probe.QContrastConfig(num_uniform_actions=4),
        generator=torch.Generator(device="cpu").manual_seed(1),
    )

    expected = {
        "q_contrast/min/var_uniform/mean",
        "q_contrast/min/range_uniform/p90",
        "q_contrast/min/policy_uniform_gap/p10",
        "q_contrast/min/grad_norm_policy/mean",
        "q_contrast/min/grad_norm_uniform/p99",
        "q_contrast/mean_ensemble/var_uniform/mean",
    }
    assert expected.issubset(metrics)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def _dataset_args(source: str):
    return SimpleNamespace(
        dataset_source=source,
        offline_dataset_path="dataset",
        offline_num_traj=3,
        reward_scale=2.0,
        reward_bias=-1.0,
        success_key="success",
        action_low=-0.5,
        action_high=0.5,
    )


def test_offline_dataset_helpers_route_minari(monkeypatch):
    from rl_garden.training.offline import _dataset

    spaces = (object(), object())
    calls = []
    monkeypatch.setattr(
        _dataset, "infer_specs_from_minari", lambda path: spaces
    )
    monkeypatch.setattr(
        _dataset,
        "load_minari_dataset_to_replay_buffer",
        lambda buffer, path, **kwargs: calls.append((buffer, path, kwargs)) or 7,
    )
    args = _dataset_args("minari")

    assert _dataset.infer_offline_dataset_specs(args) is spaces
    assert _dataset.load_offline_dataset("buffer", args) == 7
    assert calls[0][2]["num_episodes"] == 3
    assert calls[0][2]["reward_scale"] == 2.0


def test_offline_dataset_helpers_route_maniskill_h5(monkeypatch):
    from rl_garden.training.offline import _dataset

    spaces = (object(), object())
    calls = []
    monkeypatch.setattr(
        _dataset,
        "infer_specs_from_h5",
        lambda path, **kwargs: calls.append(("infer", path, kwargs)) or spaces,
    )
    monkeypatch.setattr(
        _dataset,
        "load_maniskill_h5_to_replay_buffer",
        lambda buffer, path, **kwargs: calls.append(("load", path, kwargs)) or 9,
    )
    args = _dataset_args("maniskill_h5")

    assert _dataset.infer_offline_dataset_specs(args) is spaces
    assert _dataset.load_offline_dataset("buffer", args) == 9
    assert calls[0][2] == {"action_low": -0.5, "action_high": 0.5}
    assert calls[1][2]["num_traj"] == 3


def test_diagnostic_agent_args_disable_compile():
    from rl_garden.training.offline.cql import CQLArgs

    diagnostic_args = probe.QContrastArgs(
        checkpoint_path="checkpoint.pt",
        offline_dataset_path="dataset.h5",
        buffer_device="cpu",
        device="cpu",
    )
    checkpoint = {"metadata": {"hyperparameters": {"use_compile": True}}}

    args = probe._agent_args(diagnostic_args, checkpoint, CQLArgs)

    assert args.use_compile is False
    assert args.device == "cpu"


def _write_demo_h5(path):
    with h5py.File(path, "w") as file:
        group = file.create_group("traj_0")
        group.create_dataset("obs", data=np.random.randn(9, 4).astype(np.float32))
        group.create_dataset(
            "actions",
            data=np.random.uniform(-1.0, 1.0, size=(8, 2)).astype(np.float32),
        )
        group.create_dataset("rewards", data=np.random.randn(8).astype(np.float32))
        dones = np.zeros(8, dtype=np.float32)
        dones[-1] = 1.0
        group.create_dataset("dones", data=dones)


def test_probe_q_contrast_cli_iql_checkpoint(tmp_path):
    dataset = tmp_path / "demo.h5"
    checkpoint_dir = tmp_path / "iql"
    output_json = tmp_path / "q_contrast.json"
    _write_demo_h5(dataset)

    subprocess.run(
        [
            sys.executable,
            "examples/pretrain_offline.py",
            "iql",
            "--offline-dataset-path",
            str(dataset),
            "--num-offline-steps",
            "2",
            "--buffer-device",
            "cpu",
            "--device",
            "cpu",
            "--log-type",
            "none",
            "--no-std-log",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--batch-size",
            "4",
            "--buffer-size",
            "32",
            "--n-critics",
            "4",
            "--critic-subsample-size",
            "2",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "tools/diagnostics/probe_q_contrast.py",
            "--checkpoint-path",
            str(checkpoint_dir / "iql_offline_pretrained.pt"),
            "--offline-dataset-path",
            str(dataset),
            "--buffer-device",
            "cpu",
            "--device",
            "cpu",
            "--probe-size",
            "4",
            "--num-uniform-actions",
            "3",
            "--output-json",
            str(output_json),
        ],
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "iql"
    assert payload["dataset_source"] == "maniskill_h5"
    assert payload["loaded_transitions"] == 8
    assert "q_contrast/min/var_uniform/mean" in payload["metrics"]
