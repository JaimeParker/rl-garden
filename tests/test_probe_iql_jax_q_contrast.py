from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnostics"
    / "probe_iql_jax_q_contrast.py"
)
_SPEC = importlib.util.spec_from_file_location("probe_iql_jax_q_contrast", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


class ToyLearner:
    def critic(self, observations, actions):
        q = -np.square(actions - observations).sum(axis=-1)
        return q + 1.0, q


def test_replay_neighbor_actions_selects_sorted_nearest_actions() -> None:
    observations = np.array([[0.0], [0.1], [1.0], [2.0]], dtype=np.float32)
    actions = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

    neighbors = probe.replay_neighbor_actions(
        observations,
        actions,
        start=0,
        stop=2,
        n=2,
    )

    assert neighbors.shape == (2, 2, 1)
    np.testing.assert_array_equal(neighbors[0, :, 0], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(neighbors[1, :, 0], np.array([1.0, 0.0]))


def test_candidate_q_contrast_samples_report_multitemperature_ess() -> None:
    observations = np.zeros((4, 2), dtype=np.float32)
    actor_actions = np.zeros((4, 2), dtype=np.float32)
    candidates = np.array([[[0.0, 0.0], [0.1, 0.0], [0.8, 0.0]]] * 4, dtype=np.float32)

    samples = probe.candidate_q_contrast_samples(
        ToyLearner(),
        observations,
        actor_actions,
        candidates,
        denominator=0.1,
        temperature_multipliers=(0.5, 1.0, 2.0),
    )

    np.testing.assert_allclose(samples["q_actor"], np.zeros(4))
    np.testing.assert_array_equal(samples["actor_top1"], np.ones(4))
    assert {"ess_x0p5", "ess_x1p0", "ess_x2p0"} <= set(samples)
    assert np.all(samples["ess_x0p5"] <= samples["ess_x2p0"])


def test_local_q_contrast_includes_anchor_in_boltzmann_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        probe,
        "_action_grad_norm",
        lambda _learner, _observations, actions: np.zeros(actions.shape[0]),
    )
    observations = np.zeros((4, 2), dtype=np.float32)
    actions = np.zeros((4, 2), dtype=np.float32)

    samples = probe.local_q_contrast_samples(
        ToyLearner(),
        observations,
        actions,
        radius=0.1,
        num_noisy_actions=8,
        denominator=0.001,
        rng=np.random.default_rng(4),
    )

    np.testing.assert_allclose(samples["q_anchor"], np.zeros(4))
    assert np.all(samples["local_ess"] < 2.0)
    assert np.all(samples["max_weight"] > 0.99)
    np.testing.assert_array_equal(samples["anchor_top1"], np.ones(4))


def test_validate_args_rejects_nonpositive_iql_temperature() -> None:
    args = probe.Args(
        env_name="antmaze-medium-diverse-v2",
        model_dir="models/0",
        temperature=0.0,
    )

    with pytest.raises(ValueError, match="--temperature"):
        probe.validate_args(args)


def test_fixed_radius_noise_reports_clipping() -> None:
    actions = np.full((4, 2), 0.99, dtype=np.float32)
    noisy, _effective_radius, clip_fraction = probe.fixed_radius_noise(
        actions,
        radius=0.5,
        n=16,
        rng=np.random.default_rng(1),
    )

    assert np.all(noisy <= 1.0)
    assert np.all(noisy >= -1.0)
    assert clip_fraction.mean() > 0
