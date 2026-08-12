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


def test_candidate_q_contrast_samples_report_multitemperature_ess_and_coverage() -> None:
    class ToyPolicy:
        def q_values_all(self, features, actions, target=False):
            del target
            q = -(actions - features).pow(2).sum(dim=-1, keepdim=True)
            return torch.stack([q + 2.0, q], dim=0)

    features = torch.zeros((4, 2))
    actor_actions = torch.zeros((4, 2))
    candidates = torch.tensor(
        [[[0.0, 0.0], [0.1, 0.0], [0.8, 0.0]]] * 4,
        dtype=torch.float32,
    )

    samples = probe.candidate_q_contrast_samples(
        ToyPolicy(),
        features,
        actor_actions,
        candidates,
        denominator=0.1,
        temperature_multipliers=(0.5, 1.0, 2.0),
    )

    assert torch.allclose(samples["q_actor"], torch.zeros(4))
    assert torch.all(samples["q_best_minus_actor"] == 0)
    assert torch.all(samples["actor_to_best_dist"] == 0)
    assert torch.all(samples["actor_top1"] == 1)
    assert set(samples) >= {"ess_x0p5", "ess_x1p0", "ess_x2p0"}
    assert torch.all(samples["ess_x0p5"] <= samples["ess_x2p0"])


def test_iql_denominator_uses_inverse_temperature() -> None:
    args = probe.Args(
        algorithm="iql",
        dataset_path="dataset.h5",
        checkpoint_path="checkpoint.pt",
        temperature=10.0,
    )

    name, value = probe._denominator(object(), args)

    assert name == "inverse_iql_temperature"
    assert value == 0.1


def test_validate_args_rejects_empty_candidate_sets() -> None:
    args = probe.Args(
        algorithm="iql",
        dataset_path="dataset.h5",
        checkpoint_path="checkpoint.pt",
        num_actor_candidates=0,
    )

    try:
        probe._validate_args(args)
    except ValueError as exc:
        assert "--num-actor-candidates" in str(exc)
    else:
        raise AssertionError("Expected invalid candidate count to fail.")


def test_replay_neighbor_actions_select_nearest_observation_actions() -> None:
    flat_obs = torch.tensor([[0.0], [0.1], [1.0], [2.0]])
    actions = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

    neighbors = probe.replay_neighbor_actions(
        flat_obs,
        actions,
        start=0,
        stop=2,
        n=2,
    )

    assert neighbors.shape == (2, 2, 1)
    assert torch.equal(neighbors[0, :, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(neighbors[1, :, 0], torch.tensor([1.0, 0.0]))


def test_load_d4rl_legacy_demo_batch_samples_flat_dataset(monkeypatch) -> None:
    class FakeEnv:
        def get_dataset(self):
            return {
                "observations": torch.arange(20, dtype=torch.float32).reshape(10, 2).numpy(),
                "actions": torch.arange(30, dtype=torch.float32).reshape(10, 3).numpy(),
                "terminals": torch.tensor(
                    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=torch.bool
                ).numpy(),
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "rl_garden.buffers.d4rl_legacy_dataset._make_legacy_env",
        lambda env_id: FakeEnv(),
    )

    batch = probe.load_demo_batch(
        "antmaze-test-v2",
        max_transitions=4,
        seed=123,
        dataset_backend="d4rl_legacy",
    )

    assert batch.num_available_transitions == 10
    assert batch.num_selected_transitions == 4
    assert batch.num_total_episodes == 3
    assert batch.obs.shape == (4, 2)
    assert batch.actions.shape == (4, 3)
    assert batch.flat_obs.shape == (4, 2)


def test_args_for_algorithm_preserves_checkpoint_config(tmp_path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "iql.pt"
    checkpoint_path.write_bytes(b"")
    (run_dir / "config.json").write_text(
        """{
          "inputs": {
            "obs_mode": "state",
            "gamma": 0.99,
            "n_critics": 2,
            "actor_use_layer_norm": false,
            "critic_use_layer_norm": false,
            "offline_dataset": "old-env-v2",
            "dataset_backend": "d4rl_legacy"
          }
        }""",
        encoding="utf-8",
    )

    args = probe.Args(
        algorithm="iql",
        dataset_path="antmaze-test-v2",
        checkpoint_path=str(checkpoint_path),
        dataset_backend="d4rl_legacy",
        gamma=0.8,
        n_critics=10,
        actor_use_layer_norm=True,
        critic_use_layer_norm=True,
    )

    algo_args = probe._args_for_algorithm(args)

    assert algo_args.obs_mode == "state"
    assert algo_args.gamma == 0.99
    assert algo_args.n_critics == 2
    assert algo_args.actor_use_layer_norm is False
    assert algo_args.critic_use_layer_norm is False
    assert algo_args.offline_dataset == "antmaze-test-v2"
    assert algo_args.dataset_backend == "d4rl_legacy"

    summary = probe._resolved_algorithm_summary(args, algo_args)
    assert summary["config_path_used"] == str(run_dir / "config.json")
    assert summary["obs_mode"] == "state"
    assert summary["gamma"] == 0.99


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
