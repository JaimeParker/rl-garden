from __future__ import annotations

import json
import subprocess
import sys

import h5py
import numpy as np
import torch

from rl_garden.common.q_contrast import (
    QContrastConfig,
    compute_action_grad_norm,
    compute_q_contrast_metrics,
    summary_stats,
    uniform_actions,
)


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
        return torch.stack([q1, q2], dim=0)


def test_summary_stats_reports_percentiles():
    stats = summary_stats(
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
    generator = torch.Generator(device="cpu").manual_seed(0)
    actions = uniform_actions(
        batch_size=3,
        num_actions=5,
        action_low=torch.tensor([-1.0, -0.5]),
        action_high=torch.tensor([1.0, 0.5]),
        generator=generator,
    )

    assert actions.shape == (3, 5, 2)
    assert torch.all(actions[..., 0] >= -1.0)
    assert torch.all(actions[..., 0] <= 1.0)
    assert torch.all(actions[..., 1] >= -0.5)
    assert torch.all(actions[..., 1] <= 0.5)


def test_compute_action_grad_norm_for_min_q():
    policy = ToyPolicy()
    obs = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    actions = torch.tensor([[0.5, -0.25], [0.1, 0.2]])

    grad_norm = compute_action_grad_norm(policy, obs, actions, ensemble_reduce="min")

    assert torch.allclose(grad_norm, torch.tensor([5.0, 2.0]))


def test_compute_q_contrast_metrics_smoke():
    policy = ToyPolicy()
    obs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    generator = torch.Generator(device="cpu").manual_seed(1)

    metrics = compute_q_contrast_metrics(
        policy,
        obs,
        action_low=torch.tensor([-1.0, -1.0]),
        action_high=torch.tensor([1.0, 1.0]),
        config=QContrastConfig(num_uniform_actions=4),
        generator=generator,
    )

    expected_keys = {
        "q_contrast/min/var_uniform/mean",
        "q_contrast/min/range_uniform/p90",
        "q_contrast/min/policy_uniform_gap/p10",
        "q_contrast/min/grad_norm_policy/mean",
        "q_contrast/min/grad_norm_uniform/p99",
        "q_contrast/mean_ensemble/var_uniform/mean",
    }
    assert expected_keys.issubset(metrics)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def _write_demo_h5(path):
    with h5py.File(path, "w") as f:
        group = f.create_group("traj_0")
        obs = np.random.randn(9, 4).astype(np.float32)
        actions = np.random.uniform(-1.0, 1.0, size=(8, 2)).astype(np.float32)
        rewards = np.random.randn(8).astype(np.float32)
        dones = np.zeros((8,), dtype=np.float32)
        dones[-1] = 1.0
        group.create_dataset("obs", data=obs)
        group.create_dataset("actions", data=actions)
        group.create_dataset("rewards", data=rewards)
        group.create_dataset("dones", data=dones)


def test_diagnose_q_contrast_cli_iql_checkpoint(tmp_path):
    dataset = tmp_path / "demo.h5"
    checkpoint_dir = tmp_path / "iql"
    output_json = tmp_path / "q_contrast.json"
    _write_demo_h5(dataset)

    train_cmd = [
        sys.executable,
        "examples/pretrain_offline.py",
        "--algorithm",
        "iql",
        "--offline_dataset_path",
        str(dataset),
        "--num_offline_steps",
        "2",
        "--buffer_device",
        "cpu",
        "--device",
        "cpu",
        "--log_type",
        "none",
        "--no-std-log",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--batch_size",
        "4",
        "--buffer_size",
        "32",
        "--n_critics",
        "4",
        "--critic_subsample_size",
        "2",
    ]
    subprocess.run(train_cmd, check=True)

    diagnose_cmd = [
        sys.executable,
        "examples/diagnose_q_contrast.py",
        "--checkpoint_path",
        str(checkpoint_dir / "iql_offline_pretrained.pt"),
        "--offline_dataset_path",
        str(dataset),
        "--buffer_device",
        "cpu",
        "--device",
        "cpu",
        "--probe_size",
        "4",
        "--num_uniform_actions",
        "3",
        "--output_json",
        str(output_json),
        "--no-std-log",
    ]
    subprocess.run(diagnose_cmd, check=True)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["algorithm"] == "iql"
    assert payload["loaded_transitions"] == 8
    assert payload["probe_size"] == 4
    assert "q_contrast/min/var_uniform/mean" in payload["metrics"]
