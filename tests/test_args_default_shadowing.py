"""Sibling Args defaults need an explicit override or a documented decision."""
from __future__ import annotations

import dataclasses

import pytest
from test_args_builder_drift import capture_builder, registered_entries

from rl_garden.common.cli_args import LoggingArgs
from rl_garden.training.offline.args import OfflineEvalArgs
from rl_garden.training.offline.entrypoints.qam import QAMArgs

ENTRIES = registered_entries()
ALLOWED_SHADOWS = {
    ("num_eval_steps", OfflineEvalArgs, LoggingArgs): (
        "Offline evaluation intentionally uses None to derive its length instead of 50."
    ),
}


def _default(field):
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


def unresolved_shadows(args_cls):
    for field in dataclasses.fields(args_cls):
        declarations = [
            cls for cls in args_cls.__mro__
            if field.name in vars(cls).get("__annotations__", {})
        ]
        winner = declarations[0]
        winning_default = _default(winner.__dataclass_fields__[field.name])
        for shadowed in declarations[1:]:
            shadowed_default = _default(shadowed.__dataclass_fields__[field.name])
            if winning_default != shadowed_default and not issubclass(winner, shadowed):
                yield (field.name, winner, shadowed), (winning_default, shadowed_default)


@pytest.mark.parametrize("key", sorted(ENTRIES), ids=["/".join(key) for key in sorted(ENTRIES)])
def test_args_defaults_have_no_unresolved_shadow(key):
    unexpected = [
        f"{field}: {winner.__name__}={values[0]!r} shadows {shadowed.__name__}={values[1]!r}"
        for (field, winner, shadowed), values in unresolved_shadows(ENTRIES[key].args_cls)
        if (field, winner, shadowed) not in ALLOWED_SHADOWS
    ]
    assert not unexpected, "\n".join(unexpected)


def test_shadow_allow_table_is_current():
    observed = {
        rule for entry in ENTRIES.values()
        for rule, _ in unresolved_shadows(entry.args_cls)
    }
    assert not (ALLOWED_SHADOWS.keys() - observed), "stale shadow allowances"
    assert all(reason.strip() for reason in ALLOWED_SHADOWS.values())


def test_qam_default_gradient_clip_reaches_constructor():
    args = QAMArgs()
    assert args.grad_clip_norm == 1.0
    _, _, captured = capture_builder("offline", "qam", ENTRIES["offline", "qam"], args=args)
    assert captured["kwargs"]["grad_clip_norm"] == 1.0
