from __future__ import annotations


def test_hitl_registry_discovers_residual_hil_serl():
    from rl_garden.training.hitl import registry

    registry.discover()

    assert "residual_hil_serl" in registry.entries()
