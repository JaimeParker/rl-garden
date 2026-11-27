"""Sibling Args defaults need an explicit override or a documented decision."""
from __future__ import annotations

import dataclasses

import pytest
from test_args_builder_drift import capture_builder, registered_entries

from rl_garden.common.cli_args import LoggingArgs
from rl_garden.hyperparameters.bppo import BPPOCriticParams, BPPOParams
from rl_garden.hyperparameters.unio4 import UniO4Params
from rl_garden.hyperparameters.shared import OfflineOptimizationArgs
from rl_garden.hyperparameters.awac import AWACParams
from rl_garden.hyperparameters.qam import QAMParams
from rl_garden.hyperparameters.qgf import QGFParams
from rl_garden.hyperparameters.bcq import BCQParams
from rl_garden.hyperparameters.plas import PLASParams
from rl_garden.hyperparameters.idql import IDQLParams

from rl_garden.hyperparameters.iql import IQLParams
from rl_garden.hyperparameters.fql import FQLParams
from rl_garden.hyperparameters.td3_bc import TD3CoreParams
from rl_garden.hyperparameters.shared import (
    OfflineDeterministicActorCriticArgs,
    OfflineActorArgs,
    OfflineCriticArgs,
)
from rl_garden.training.offline.args import OfflineDiscountArgs, OfflineValueArgs
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


# Equal-valued duplicates need their own table: they are not unresolved shadows.
# Explicit subclass overrides (including the leaf's family presets) are intentional.

ALLOWED_PARAM_DUPLICATES = {
    ("tau", BPPOCriticParams, OfflineDiscountArgs):
        "The shared critic owns tau; the discount mixin also supplies gamma and utd.",
    **{
        (name, BPPOCriticParams, OfflineOptimizationArgs):
        "The shared critic owns optimizer settings also present in the legacy run mixin."
        for name in ("weight_decay", "use_adamw")
    },
    **{
        ("grad_clip_norm", params, OfflineOptimizationArgs):
        "The actor owns gradient clipping also present in the legacy run mixin."
        for params in (BPPOParams, UniO4Params)
    },
    **{
        (name, params, OfflineActorArgs):
        "The actor owns network settings also present in the legacy actor CLI mixin."
        for params in (BPPOParams, UniO4Params)
        for name in ("num_groups", "kernel_init", "backbone_type", "std_parameterization")
    },
    ("value_use_layer_norm", IDQLParams, OfflineValueArgs):
        "IDQL owns its value normalization; the legacy mixin also supplies unused CLI flags.",
    **{
        ("tau", params, OfflineDiscountArgs):
        "The algorithm owns its target update rate; the loop mixin also supplies gamma and utd."
        for params in (QAMParams, QGFParams, BCQParams, PLASParams, IDQLParams)
    },
    ("tau", FQLParams, OfflineDiscountArgs):
        "FQL owns its target update rate; the loop mixin also supplies gamma and utd.",
    ("tau", OfflineDiscountArgs, IQLParams):
        "The off2on loop mixin and IQL share the same target update rate.",
    ("tau", IQLParams, OfflineDiscountArgs):
        "IQL owns its target update rate; the discount mixin also supplies gamma and utd.",
    ("kernel_init", IQLParams, OfflineActorArgs):
        "IQL owns the supported initializer type; the off2on actor mixin shares its default.",
    ("std_parameterization", IQLParams, OfflineActorArgs):
        "IQL and the off2on actor mixin share the Gaussian scale parameterization.",
    ("tau", TD3CoreParams, OfflineDiscountArgs):
        "TD3-BC owns its target update rate; the discount mixin also supplies gamma and utd.",
    ("kernel_init", TD3CoreParams, OfflineActorArgs):
        "TD3-BC owns its initializer choices; off2on also inherits the shared actor mixin.",
    ("tau", AWACParams, OfflineDiscountArgs):
        "AWAC owns its target update rate; the shared discount mixin also supplies gamma and utd.",
    **{
        (name, OfflineDeterministicActorCriticArgs, OfflineActorArgs):
        "The deterministic network mixin and the off2on Gaussian actor mixin share these toggles."
        for name in ("actor_use_group_norm", "num_groups", "actor_dropout_rate", "backbone_type")
    },
    **{
        (name, OfflineDeterministicActorCriticArgs, OfflineCriticArgs):
        "The fixed-critic network mixin and the off2on ensemble mixin share these toggles."
        for name in ("critic_use_group_norm", "critic_dropout_rate")
    },
    **{
        (name, AWACParams, OfflineActorArgs):
        "AWAC owns its Gaussian actor choices; off2on also inherits the shared actor mixin."
        for name in ("kernel_init", "std_parameterization")
    },
}


def test_parameter_duplicate_allowances_are_exact():
    observed = set()
    for key in (("online", "flash_sac"), ("offline", "edac"),
    ("offline", "cql"), ("offline", "calql"), ("offline", "wsrl"),
    ("off2on", "calql"), ("off2on", "wsrl"),
                ("online", "dreamer_v3"), ("online", "ncrl"), ("offline", "rssm_pretrain"),
                *(("online", name) for name in (
                    "sac", "sac_flow", "recurrent_sac", "transformer_sac",
                    "rlpd", "rlpd_hybrid", "explore", "supe", "acrlpd")),
                ("online", "tdmpc2"), ("online", "supermimic"),
                ("offline", "tdmpc2_multitask"), ("offline", "pwm"),
                *(("online", name) for name in (
                    "ppo", "recurrent_ppo", "transformer_ppo", "gail", "dppo", "flow_ppo",
                    "diffusion_cm_distill_online", "policy_distillation")),
                ("online", "dagger"), ("online", "drqv2"), ("online", "td3"),
                ("offline", "hilp"), ("offline", "opal"),
                *(("offline", name) for name in ("bppo", "unio4", "unio4_ope")),
                *(("offline", name) for name in
                  ("bc", "flow_bc", "mean_flow_bc", "diffusion_bc", "a2a_bc", "consistency_distill_bc")),
                *(("offline", name) for name in ("qam", "qgf", "bcq", "plas", "idql")),
                ("offline", "iql"), ("off2on", "iql"),
                ("offline", "awac"), ("off2on", "awac"),
                ("offline", "td3_bc"), ("offline", "rebrac"),
                ("offline", "spot"), ("off2on", "spot"),
                *((phase, name) for phase in ("offline", "off2on")
                  for name in ("fql", "floq", "fino", "fac", "value_flows", "acfql")
                  if (phase, name) in ENTRIES)):
        entry = ENTRIES[key]
        for field in dataclasses.fields(entry.algorithm_cls().params_class):
            declarations = [cls for cls in entry.args_cls.__mro__
                            if field.name in vars(cls).get("__annotations__", {})]
            winner = declarations[0]
            observed.update((field.name, winner, shadowed) for shadowed in declarations[1:]
                            if not issubclass(winner, shadowed))
    assert observed == ALLOWED_PARAM_DUPLICATES.keys()
    assert all(reason.strip() for reason in ALLOWED_PARAM_DUPLICATES.values())


# This leaf uses an omission sentinel while its constructor keeps a numeric default.
ALLOWED_DREAMER_LEAF_DEFAULTS = {
    ("discount_horizon", "DreamerV3TrainingArgs"): "None means use the legacy --horizon value.",
}


def test_dreamer_leaf_default_allowance_is_exact():
    from rl_garden.hyperparameters.dreamer import DreamerV3Params
    from rl_garden.training.online.args import DreamerV3TrainingArgs

    observed = {
        (field.name, DreamerV3TrainingArgs.__name__)
        for field in dataclasses.fields(DreamerV3Params)
        if field.name in DreamerV3TrainingArgs.__annotations__
        and getattr(DreamerV3TrainingArgs, field.name) != field.default
    }
    assert observed == ALLOWED_DREAMER_LEAF_DEFAULTS.keys()
    assert all(ALLOWED_DREAMER_LEAF_DEFAULTS.values())


ALLOWED_OFFLINE_ALIAS_LEAVES = {
    **{("offline", name, field): "None uses --actor-" + field.replace("_", "-")
       for name in ("bppo", "unio4", "unio4_ope")
       for field in ("use_layer_norm", "use_group_norm", "dropout_rate")},
    ("offline", "a2a_bc", "cnn_activation_fn"): "None uses --activation-fn",
}


def test_offline_alias_leaf_allowance_is_exact():
    from typing import get_args, get_type_hints

    observed = set()
    for phase, name in {key[:2] for key in ALLOWED_OFFLINE_ALIAS_LEAVES}:
        entry = ENTRIES[phase, name]
        hints = get_type_hints(entry.args_cls)
        for field in dataclasses.fields(entry.algorithm_cls().params_class):
            if (field.name in entry.args_cls.__annotations__
                    and getattr(entry.args_cls, field.name) is None):
                observed.add((phase, name, field.name))
                assert type(None) in get_args(hints[field.name])
    assert observed == ALLOWED_OFFLINE_ALIAS_LEAVES.keys()
    assert all(ALLOWED_OFFLINE_ALIAS_LEAVES.values())
