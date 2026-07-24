from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnostics"
    / "probe_q_contrast_local_noise.py"
)
_SPEC = importlib.util.spec_from_file_location("probe_q_contrast_local_noise", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_fixed_radius_noise_preserves_radius_when_unclipped() -> None:
    actions = torch.zeros((8, 3))
    low = torch.full((3,), -1.0)
    high = torch.full((3,), 1.0)
    generator = torch.Generator().manual_seed(123)

    noisy, effective_radius, clip_fraction = probe.fixed_radius_noise(
        actions,
        radius=0.2,
        n=5,
        low=low,
        high=high,
        generator=generator,
    )

    assert noisy.shape == (8, 5, 3)
    assert torch.allclose(effective_radius, torch.full((8, 5), 0.2), atol=1e-6)
    assert torch.count_nonzero(clip_fraction) == 0


def test_fixed_radius_noise_reports_clipping() -> None:
    actions = torch.full((4, 2), 0.99)
    low = torch.full((2,), -1.0)
    high = torch.full((2,), 1.0)
    generator = torch.Generator().manual_seed(1)

    noisy, _effective_radius, clip_fraction = probe.fixed_radius_noise(
        actions,
        radius=0.5,
        n=16,
        low=low,
        high=high,
        generator=generator,
    )

    assert torch.all(noisy <= 1.0)
    assert torch.all(noisy >= -1.0)
    assert clip_fraction.mean() > 0


def test_local_q_contrast_samples_use_min_q_and_boltzmann_metrics() -> None:
    class ToyPolicy:
        def q_values_all(self, features, actions, target=False):
            del target
            base_q = -(actions - features).pow(2).sum(dim=-1, keepdim=True)
            return torch.stack([base_q + 1.0, base_q], dim=0)

    features = torch.zeros((6, 2))
    actions = torch.zeros((6, 2))
    low = torch.full((2,), -1.0)
    high = torch.full((2,), 1.0)
    generator = torch.Generator().manual_seed(4)

    samples = probe.local_q_contrast_samples(
        ToyPolicy(),
        features,
        actions,
        radius=0.1,
        num_noisy_actions=8,
        denominator=0.001,
        action_low=low,
        action_high=high,
        generator=generator,
    )

    assert torch.allclose(samples["q_anchor"], torch.zeros(6))
    assert torch.all(samples["q_drop"] > 0)
    assert torch.allclose(samples["local_lipschitz"], torch.full((6,), 0.1), atol=1e-5)
    assert torch.all(samples["dq_over_denominator"] > 0)
    assert torch.all(samples["local_ess"] < 2.0)
    assert torch.all(samples["anchor_top1"] == 1.0)
    assert torch.allclose(samples["action_grad_norm"], torch.zeros(6))


def test_summarize_samples_reports_requested_quantiles() -> None:
    summary = probe.summarize_samples(
        {
            "q_drop": [torch.tensor([1.0, 2.0, 3.0])],
            "local_ess": [torch.tensor([1.0, 2.0, 3.0])],
            "max_weight": [torch.tensor([0.2, 0.5, 0.9])],
            "action_grad_norm": [torch.tensor([1.0, 2.0, 100.0])],
        }
    )

    assert summary["q_drop_mean"] == 2.0
    assert summary["q_drop_p50"] == 2.0
    assert summary["local_ess_p10"] == torch.quantile(
        torch.tensor([1.0, 2.0, 3.0]), 0.1
    ).item()
    assert summary["max_weight_p90"] == torch.quantile(
        torch.tensor([0.2, 0.5, 0.9]), 0.9
    ).item()
    assert summary["action_grad_norm_p99"] == torch.quantile(
        torch.tensor([1.0, 2.0, 100.0]), 0.99
    ).item()
