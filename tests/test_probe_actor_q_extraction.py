from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from gymnasium import spaces

from rl_garden.encoders.combined import CombinedExtractor, default_image_encoder_factory
from rl_garden.encoders.plain_conv import PlainConv


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "diagnostics"
    / "probe_actor_q_extraction.py"
)
_SPEC = importlib.util.spec_from_file_location("probe_actor_q_extraction", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def _write_traj(handle, name: str, *, success: bool, action_value: float) -> None:
    traj = handle.create_group(name)
    traj.attrs["final_success"] = success
    obs = traj.create_group("obs")
    obs.create_dataset("rgb_base_camera", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    obs.create_dataset("state", data=np.zeros((3, 5), dtype=np.float32))
    traj.create_dataset("actions", data=np.full((3, 2), action_value, dtype=np.float32))


def test_load_demo_batch_filters_success_episodes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo.h5"
    with h5py.File(dataset_path, "w") as handle:
        _write_traj(handle, "traj_0", success=False, action_value=-1.0)
        _write_traj(handle, "traj_1", success=True, action_value=2.0)

    all_batch = probe.load_demo_batch(
        dataset_path, scope="all", max_transitions=0, seed=1
    )
    success_batch = probe.load_demo_batch(
        dataset_path, scope="success", max_transitions=0, seed=1
    )

    assert all_batch.actions.shape == (6, 2)
    assert success_batch.actions.shape == (3, 2)
    assert torch.all(success_batch.actions == 2.0)
    assert success_batch.num_available_transitions == 3
    assert success_batch.num_selected_transitions == 3
    assert success_batch.num_success_episodes == 1
    assert success_batch.num_total_episodes == 2


def test_relative_dormant_ratio_uses_layer_relative_scale() -> None:
    activations = torch.tensor(
        [
            [10.0, 1.0, 0.01],
            [10.0, 1.0, 0.01],
        ]
    )

    assert torch.isclose(
        probe.relative_dormant_ratio(activations, threshold=0.1), torch.tensor(1 / 3)
    )


def test_q_and_grad_uses_sum_scaled_action_gradient() -> None:
    class ToyPolicy:
        def q_values_all(self, features, actions, target=False):
            del target
            q = (features + actions).sum(dim=-1, keepdim=True)
            return torch.stack([q, q + 1.0], dim=0)

    features = torch.zeros((4, 2))
    actions = torch.zeros((4, 2))
    q, grad_norm = probe._q_and_grad(ToyPolicy(), features, actions)

    assert q.shape == (4, 1)
    assert torch.allclose(q, torch.zeros_like(q))
    assert torch.isclose(grad_norm, torch.sqrt(torch.tensor(2.0)))


def test_plain_conv_activation_probe_reports_gap_bottleneck_and_gradients() -> None:
    encoder = PlainConv(
        spaces.Box(0.0, 1.0, (3, 64, 64), dtype=np.float32),
        features_dim=16,
        image_size=(64, 64),
        pooling="gap",
    )
    image = torch.randn(2, 3, 64, 64)

    with probe.PlainConvActivationProbe(encoder) as activation_probe:
        encoded = encoder(image)
    metrics = activation_probe.metrics(encoded.sum())[""]

    assert metrics["pooling"] == "gap"
    assert metrics["pre_pool_shape"] == [64, 4, 4]
    assert metrics["flatten_input_dim"] == 1024
    assert metrics["bottleneck_input_dim"] == 64
    assert metrics["bottleneck_output_dim"] == 16
    assert metrics["bottleneck_parameter_count"] == 1_040
    assert metrics["bottleneck_compression_ratio"] == 16.0
    assert metrics["q_grad_pre_pool_norm"] > 0
    assert metrics["q_grad_pooled_norm"] > 0
    assert metrics["q_grad_encoded_norm"] > 0


def test_plain_conv_activation_probe_reports_each_per_key_camera() -> None:
    obs_space = spaces.Dict(
        {
            "rgb_base_camera": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            "rgb_hand_camera": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        }
    )
    extractor = CombinedExtractor(
        obs_space,
        image_keys=("rgb_base_camera", "rgb_hand_camera"),
        image_encoder_factory=default_image_encoder_factory(
            features_dim=16,
            plain_conv_pooling="gap",
        ),
        use_proprio=False,
        fusion_mode="per_key",
    )
    obs = {
        "rgb_base_camera": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "rgb_hand_camera": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
    }

    with probe.PlainConvActivationProbe(extractor) as activation_probe:
        encoded = extractor(obs)
    metrics = activation_probe.metrics(encoded.sum())

    assert set(metrics) == {
        "image_encoders.rgb_base_camera",
        "image_encoders.rgb_hand_camera",
    }
    assert all(branch["pooling"] == "gap" for branch in metrics.values())


def test_probe_checkpoint_isolates_process_rng(monkeypatch) -> None:
    class FakeAgent:
        device = torch.device("cpu")

    def consume_rng(*args, **kwargs):
        del args, kwargs
        torch.rand(4)
        return {"ok": True}

    monkeypatch.setattr(probe, "_probe_checkpoint_impl", consume_rng)
    torch.manual_seed(123)
    state_before = torch.get_rng_state()

    result = probe.probe_checkpoint(
        FakeAgent(),
        batch=None,
        include_q=True,
        seed=1,
        batch_size=1,
    )

    assert result == {"ok": True}
    assert torch.equal(torch.get_rng_state(), state_before)
