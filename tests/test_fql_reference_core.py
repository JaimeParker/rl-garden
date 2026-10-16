"""Focused invariants for FQL's opt-in JAX-reference execution path."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden.algorithms import FQL, Off2OnFQL, OfflineEnvSpec
from rl_garden.encoders.config import EncoderConfig
from rl_garden.policies._fql_networks import FQLMLP


@pytest.mark.parametrize(
    ("module_name", "args_name", "kwargs_name"),
    [
        ("rl_garden.training.offline.fql", "FQLArgs", "_fql_kwargs"),
        ("rl_garden.training.offline.floq", "FloQArgs", "_floq_kwargs"),
        ("rl_garden.training.offline.fino", "FINOArgs", "_fino_kwargs"),
        ("rl_garden.training.offline.fac", "FACArgs", "_fac_kwargs"),
        (
            "rl_garden.training.offline.value_flows",
            "ValueFlowsArgs",
            "_value_flows_kwargs",
        ),
    ],
)
def test_fql_family_inherits_and_forwards_shared_net_arch(
    module_name, args_name, kwargs_name
):
    from rl_garden.training.offline._args import OfflineFQLArgs

    module = importlib.import_module(module_name)
    args_class = getattr(module, args_name)
    kwargs_builder = getattr(module, kwargs_name)

    assert "net_arch" in OfflineFQLArgs.__annotations__
    assert get_type_hints(OfflineFQLArgs)["net_arch"] == tuple[int, ...]
    if args_name == "FQLArgs":
        assert "net_arch" not in args_class.__annotations__

    default_args = args_class()
    custom_args = args_class(net_arch=(64, 32))
    assert default_args.net_arch == (512, 512, 512, 512)
    assert custom_args.net_arch == (64, 32)
    assert kwargs_builder(default_args, None, None)["net_arch"] == [512, 512, 512, 512]
    assert kwargs_builder(custom_args, None, None)["net_arch"] == [64, 32]


def _agent(**overrides) -> FQL:
    env = OfflineEnvSpec(
        spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    defaults = {
        "env": env, "buffer_size": 64, "buffer_device": "cpu", "batch_size": 8, "device": "cpu",
        "net_arch": [8, 8], "flow_steps": 3, "tau": 0.5, "reference_compatible": True,
    }
    defaults.update(overrides)
    agent = FQL(**defaults)
    for _ in range(16):
        agent.replay_buffer.add(
            {"state": torch.randn(1, 4)}, {"state": torch.randn(1, 4)},
            torch.rand(1, 2) * 2 - 1, torch.randn(1), torch.zeros(1),
        )
    return agent


def _noise() -> dict[str, torch.Tensor]:
    return {
        "next_noise": torch.randn(8, 2), "x0": torch.randn(8, 2),
        "times": torch.rand(8, 1), "distill_noise": torch.randn(8, 2),
    }


def test_fql_mlp_uses_tanh_gelu_then_flax_epsilon_layer_norm():
    mlp = FQLMLP(2, [2], 1, layer_norm=True)
    assert mlp.layer_norms[0].eps == pytest.approx(1e-6)
    with torch.no_grad():
        mlp.linears[0].weight.copy_(torch.eye(2))
        mlp.linears[0].bias.zero_()
        mlp.linears[1].weight.fill_(1.0)
        mlp.linears[1].bias.zero_()
    x = torch.tensor([[1.0, -0.5]])
    hidden = torch.nn.functional.gelu(x, approximate="tanh")
    expected = mlp.linears[1](mlp.layer_norms[0](hidden))
    assert torch.allclose(mlp(x), expected)


def test_reference_losses_and_update_are_deterministic_for_supplied_noise():
    agent = _agent()
    data = agent.replay_buffer.sample(8)
    noises = _noise()
    first = agent.reference_losses(data, **noises)
    second = agent.reference_losses(data, **noises)
    for key in first:
        assert torch.allclose(first[key], second[key]), key
    metrics = agent.reference_update(data, **noises)
    assert set(metrics) == {"critic_loss", "actor_loss", "bc_flow_loss", "distill_loss", "q_loss"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_reference_target_ema_uses_pre_update_critic():
    agent = _agent()
    data = agent.replay_buffer.sample(8)
    critic_before = [p.detach().clone() for p in agent.policy.critic.parameters()]
    target_before = [p.detach().clone() for p in agent.policy.critic_target.parameters()]
    agent.reference_update(data, **_noise())
    for target, critic, old_target in zip(agent.policy.critic_target.parameters(), critic_before, target_before):
        assert torch.allclose(target, old_target * 0.5 + critic * 0.5)


def test_reference_mode_rejects_visual_recipe():
    with pytest.raises(ValueError, match="state MLP recipe"):
        _agent(encoder_config=EncoderConfig(features_dim=8))


def test_reference_mode_rejects_image_schema_without_explicit_encoder_config():
    visual_env = OfflineEnvSpec(
        spaces.Dict(
            {
                "state": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "rgb_front": spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8),
            }
        ),
        spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="visual/encoded observations"):
        _agent(env=visual_env, encoder_config=None)


def test_offline_builder_configures_precision_only_for_reference_recipe(monkeypatch):
    from rl_garden.training import inspection
    from rl_garden.training.offline import fql as offline_fql

    precision: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precision.append)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(offline_fql, "_fql_kwargs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(inspection, "construct_agent", lambda *_args, **_kwargs: object())

    offline_fql.build_fql(SimpleNamespace(reference_compatible=False), None, None)
    assert precision == []
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True

    offline_fql.build_fql(SimpleNamespace(reference_compatible=True), None, None)
    assert precision == ["highest"]
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_fql_constructors_do_not_mutate_precision_flags():
    previous_precision = torch.get_float32_matmul_precision()
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _agent(reference_compatible=False)
        _agent(reference_compatible=True)
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


@pytest.mark.parametrize("strict", [True, False])
def test_checkpoint_rejects_reference_mode_mismatch_before_state_mutation(tmp_path, strict):
    reference = _agent()
    path = reference.save(tmp_path / "reference.pt")
    legacy = _agent(reference_compatible=False)
    legacy._global_step = 17
    policy_before = {
        name: parameter.detach().clone()
        for name, parameter in legacy.policy.named_parameters()
    }
    with pytest.raises(ValueError, match="reference_compatible mismatch"):
        legacy.load(path, strict=strict)
    assert legacy._global_step == 17
    assert all(
        torch.equal(parameter, policy_before[name])
        for name, parameter in legacy.policy.named_parameters()
    )


def test_reference_checkpoint_restores_targets_optimizers_and_counters(tmp_path):
    source = _agent()
    source.train(1)
    source._global_step = 17
    path = source.save(tmp_path / "reference.pt")
    restored = _agent()
    restored.load(path)
    assert restored._global_step == 17
    assert restored._global_update == source._global_update
    for source_param, restored_param in zip(
        source.policy.critic_target.parameters(), restored.policy.critic_target.parameters()
    ):
        assert torch.equal(source_param, restored_param)
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]


def test_reference_fql_checkpoint_loads_into_off2on_shell(tmp_path):
    source = _agent()
    source.train(1)
    path = source.save(tmp_path / "reference.pt")
    target = Off2OnFQL(
        source.env,
        buffer_size=64,
        buffer_device="cpu",
        batch_size=8,
        device="cpu",
        net_arch=[8, 8],
        flow_steps=3,
        tau=0.5,
        reference_compatible=True,
    )
    target.load(path)
    for source_param, target_param in zip(source.policy.critic.parameters(), target.policy.critic.parameters()):
        assert torch.equal(source_param, target_param)
