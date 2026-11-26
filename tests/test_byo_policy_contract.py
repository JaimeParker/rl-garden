"""Bring-your-own-policy from a package outside ``rl_garden`` (the actor
contracts in ``rl_garden/policies/contracts.py``).

The downstream fixture package ``rlg_downstream_example.byo_policy``
(``tests/fixtures/downstream_extension``) supplies a PPO actor head, a SAC
actor head and a feature extractor. Here they are trained by the in-tree PPO
and SAC through ``actor_factory`` / ``policy_kwargs``, wired through the
training CLI (``--actor-factory``, ``--actor-extractor``), recorded in the
checkpoint as import paths and rebuilt by evaluation from the checkpoint. CPU
only.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch
from _decision_fakes import _FixedHoldVecEnv
from test_cql_calql import _offline_env, _offline_kwargs
from test_graph_losses import _learn, _make_agent
from test_jsrl import _make_calql_checkpoint
from test_optimizer_ownership import COMMON_KWARGS, TinyVecEnv
from test_ppo_actor_factory import assert_first_ratio_is_one, record_policy_ratios
from test_sac_optimizer_snapshot import fill_random_transitions
from torch import nn

from rl_garden.policies.contracts import PPOActorCapabilities

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "downstream_extension"
PACKAGE = "rlg_downstream_example"
MODULE = f"{PACKAGE}.byo_policy"
PPO_FACTORY = f"{MODULE}:make_tanh_gaussian_ppo_actor"
SAC_FACTORY = f"{MODULE}:make_squashed_sac_actor"
EXTRACTOR = f"{MODULE}:StateBackboneExtractor"


@pytest.fixture
def byo(monkeypatch):
    """The fixture package's ``byo_policy`` module, imported afresh; CUDA
    hidden so every agent is on CPU."""
    monkeypatch.syspath_prepend(str(FIXTURE_ROOT))
    for name in [name for name in sys.modules if name.split(".")[0] == PACKAGE]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return importlib.import_module(MODULE)


def _parameters(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


# --- Python API --------------------------------------------------------------------------------


def test_ppo_trains_a_downstream_actor_head(byo):
    from rl_garden.algorithms import PPO

    agent = _make_agent(PPO, actor_factory=byo.make_tanh_gaussian_ppo_actor)
    assert isinstance(agent.policy.actor, byo.TanhGaussianPPOActor)
    assert agent.policy.actor_capabilities == PPOActorCapabilities(
        has_entropy=False, supports_per_dim_log_prob=True, is_gaussian=False
    )
    before = _parameters(agent.policy.actor)
    ratios = record_policy_ratios(agent)

    (metrics,) = _learn(agent)

    assert_first_ratio_is_one(ratios)
    assert "entropy" not in metrics
    assert all(math.isfinite(metrics[key]) for key in ("policy_loss", "value_loss", "approx_kl"))
    assert _changed(before, agent.policy.actor)


def test_sac_trains_a_downstream_actor_head(byo):
    from rl_garden.algorithms import SAC

    agent = SAC(env=TinyVecEnv(), actor_factory=byo.make_squashed_sac_actor, **COMMON_KWARGS)
    fill_random_transitions(agent)
    assert isinstance(agent.policy.actor, byo.SquashedSACActor)
    before = _parameters(agent.policy.actor)

    info = agent.train(1, compute_info=True)

    assert math.isfinite(info["critic_loss"]) and math.isfinite(info["actor_loss"])
    # SACActorBase's default diagnostics.
    assert {"log_prob_mean", "log_prob_std", "action_abs_mean"} <= set(info)
    assert _changed(before, agent.policy.actor)


class _NoClampPPOActor(nn.Module):
    def __init__(self, features_dim, action_space):
        super().__init__()
        self.mean = nn.Linear(features_dim, action_space.shape[0])

    def action_log_prob(self, features, deterministic=False, *, sum_dims=True):
        action = self.mean(features)
        return action, action.new_zeros(action.shape[0], 1), None

    def evaluate_action_log_prob(self, features, actions, *, sum_dims=True):
        return actions.new_zeros(actions.shape[0], 1), None

    def deterministic_action(self, features):
        return self.mean(features)


class _NoDeterministicSACActor(nn.Module):
    stochastic = True
    log_prob_is_density = True
    supports_evaluate_log_prob = False

    def __init__(self, features_dim, action_space):
        super().__init__()
        self.mean = nn.Linear(features_dim, action_space.shape[0])

    def action_log_prob(self, features):
        action = torch.tanh(self.mean(features))
        return action, action.new_zeros(action.shape[0], 1)


def test_an_actor_missing_a_contract_method_fails_at_construction():
    from rl_garden.algorithms import PPO, SAC

    with pytest.raises(TypeError, match="clamp_action"):
        _make_agent(PPO, actor_factory=_NoClampPPOActor)
    with pytest.raises(TypeError, match="deterministic_action"):
        SAC(env=TinyVecEnv(), actor_factory=_NoDeterministicSACActor, **COMMON_KWARGS)


def test_importance_sampled_cql_rejects_the_deterministic_actor():
    """``DeterministicTanhActor``'s zero log-prob is not a density."""
    from rl_garden.algorithms import CQL
    from rl_garden.networks import DeterministicTanhActor

    def deterministic(features_dim, action_space):
        return DeterministicTanhActor(features_dim, action_space, [8])

    with pytest.raises(ValueError, match="cql_importance_sample=True.*log_prob_is_density=False"):
        CQL(env=_offline_env(), actor_factory=deterministic, ent_coef=0.0, **_offline_kwargs())
    agent = CQL(
        env=_offline_env(),
        actor_factory=deterministic,
        ent_coef=0.0,
        cql_importance_sample=False,
        **_offline_kwargs(),
    )
    assert not agent.policy.actor_capabilities.log_prob_is_density


def test_acrlpd_rejects_an_actor_q_aggregation():
    from rl_garden.algorithms.acrlpd import ACRLPD

    with pytest.raises(ValueError, match="actor_q_agg='mean'.*q_agg"):
        ACRLPD(env=TinyVecEnv(), actor_q_agg="mean", **COMMON_KWARGS)


def test_acrlpd_rejects_an_initial_training_phase():
    from rl_garden.algorithms.acrlpd import ACRLPD
    from rl_garden.common.training_phase import InitialTrainingPhase

    phase = InitialTrainingPhase(
        duration_steps=30, update_actor=False, update_critic=True, update_encoder=False
    )
    with pytest.raises(ValueError, match="--critic-only-steps.*Q-chunking has no critic-only phase"):
        ACRLPD(env=TinyVecEnv(), initial_training_phase=phase, **COMMON_KWARGS)


def test_jsrl_rebuilds_a_guide_trained_with_an_actor_factory(byo, tmp_path, caplog):
    import logging

    from rl_garden.algorithms import jsrl

    path, agent = _make_calql_checkpoint(tmp_path, actor_factory=byo.make_squashed_sac_actor)
    with caplog.at_level(logging.INFO, logger=jsrl.__name__):
        guide = jsrl._load_guide_policy(
            path,
            "calql",
            agent.env.single_observation_space,
            agent.env.single_action_space,
            device=torch.device("cpu"),
        )
    assert f"importing actor_factory {SAC_FACTORY} from checkpoint metadata" in caplog.text

    assert isinstance(guide.actor, byo.SquashedSACActor)
    obs = {"state": torch.randn(4, *agent.env.single_observation_space["state"].shape)}
    assert torch.allclose(
        guide.predict(obs, deterministic=True), agent.policy.predict(obs, deterministic=True)
    )


# --- CLI wiring --------------------------------------------------------------------------------


def _args_classes():
    from rl_garden.training.off2on.entrypoints.calql import CalQLOff2OnArgs
    from rl_garden.training.off2on.entrypoints.so2 import SO2Off2OnArgs
    from rl_garden.training.off2on.entrypoints.wsrl import WSRLOff2OnArgs
    from rl_garden.training.offline.entrypoints.calql import CalQLArgs
    from rl_garden.training.offline.entrypoints.cql import CQLArgs
    from rl_garden.training.offline.entrypoints.edac import EDACArgs
    from rl_garden.training.offline.entrypoints.wsrl import WSRLOfflineArgs
    from rl_garden.training.online.entrypoints.acrlpd import ACRLPDArgs
    from rl_garden.training.online.entrypoints.gail import GAILArgs
    from rl_garden.training.online.entrypoints.jsrl import JSRLArgs
    from rl_garden.training.online.entrypoints.ppo import PPOArgs
    from rl_garden.training.online.entrypoints.recurrent_ppo import RecurrentPPOArgs
    from rl_garden.training.online.entrypoints.recurrent_sac import RecurrentSACArgs
    from rl_garden.training.online.entrypoints.rlpd import RLPDArgs
    from rl_garden.training.online.entrypoints.rlpd_hybrid import RLPDHybridArgs
    from rl_garden.training.online.entrypoints.sac import SACArgs
    from rl_garden.training.online.entrypoints.sac_flow import SACFlowArgs
    from rl_garden.training.online.entrypoints.transformer_ppo import TransformerPPOArgs
    from rl_garden.training.online.entrypoints.transformer_sac import TransformerSACArgs

    accepting = [
        PPOArgs, RecurrentPPOArgs, TransformerPPOArgs, GAILArgs,
        SACArgs, RLPDArgs, RLPDHybridArgs, ACRLPDArgs, JSRLArgs,
        CQLArgs, CalQLArgs, WSRLOfflineArgs, EDACArgs,
        CalQLOff2OnArgs, WSRLOff2OnArgs, SO2Off2OnArgs,
    ]
    # Their algorithms reject an actor factory (SACFlowPolicy, SequenceSAC).
    rejecting = [SACFlowArgs, RecurrentSACArgs, TransformerSACArgs]
    return accepting, rejecting


def test_policy_module_flags_sit_on_the_args_whose_algorithm_accepts_them():
    from rl_garden.common.cli_args import PolicyModuleArgs

    accepting, rejecting = _args_classes()
    for args_cls in accepting:
        assert issubclass(args_cls, PolicyModuleArgs), args_cls.__name__
    for args_cls in rejecting:
        names = {field.name for field in fields(args_cls)}
        assert "actor_factory" not in names, args_cls.__name__


def test_offline_and_off2on_builders_forward_the_resolved_modules(byo):
    from rl_garden.common import Logger
    from rl_garden.training.offline.entrypoints.cql import CQLArgs, _cql_kwargs

    args = CQLArgs(offline_dataset="demo.h5", actor_factory=SAC_FACTORY, actor_extractor=EXTRACTOR)
    kwargs = _cql_kwargs(args, _offline_env(), Logger(log_type="none"))

    assert kwargs["actor_factory"] is byo.make_squashed_sac_actor
    assert kwargs["policy_kwargs"] == {"actor_extractor_class": byo.StateBackboneExtractor}
    assert "actor_factory" not in _cql_kwargs(
        CQLArgs(offline_dataset="demo.h5"), _offline_env(), Logger(log_type="none")
    )


def test_an_unimportable_path_is_a_configuration_error(byo):
    from rl_garden.common.cli_args import PolicyModuleArgs, policy_module_kwargs
    from rl_garden.common.effective_config import ConfigError

    with pytest.raises(ConfigError, match="--actor-factory .*no_such_factory"):
        policy_module_kwargs(PolicyModuleArgs(actor_factory=f"{MODULE}:no_such_factory"))
    with pytest.raises(ConfigError, match="--critic-extractor .*no_such_module"):
        policy_module_kwargs(PolicyModuleArgs(critic_extractor="no_such_module:Extractor"))


def test_sac_args_carry_the_entropy_off_options():
    from rl_garden.training.online import registry
    from rl_garden.training.online.entrypoints.sac import sac_common_kwargs

    default = sac_common_kwargs(registry.parse_args(["sac"]), None, None, None, None, {})
    assert (default["target_policy_noise"], default["target_noise_clip"], default["actor_q_agg"]) == (
        0.0, 0.5, "min",
    )
    args = registry.parse_args(
        ["sac", "--target-policy-noise", "0.2", "--target-noise-clip", "0.3", "--actor-q-agg", "first"]
    )
    kwargs = sac_common_kwargs(args, None, None, None, None, {})
    assert (kwargs["target_policy_noise"], kwargs["target_noise_clip"], kwargs["actor_q_agg"]) == (
        0.2, 0.3, "first",
    )


def test_sac_dry_run_records_the_import_paths(byo, monkeypatch, capsys, tmp_path):
    import gymnasium as gym

    from rl_garden.training.online import registry
    from rl_garden.training.online import runner as online_runner

    class FakeEnv:
        num_envs = 1
        single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
        single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))

        def close(self):
            pass

    monkeypatch.setattr(online_runner, "make_training_envs", lambda backend, req: (FakeEnv(), None))
    common = [
        "--buffer-device", "cpu", "--buffer-size", "16", "--batch-size", "2",
        "--hidden-dim", "8", "--checkpoint-dir", str(tmp_path / "checkpoints"),
        "--log-type", "none",
    ]
    modules = ["--actor-factory", SAC_FACTORY, "--actor-extractor", EXTRACTOR]

    registry.run_cli(["sac", "--print-config", *common, *modules])
    printed = capsys.readouterr().out
    assert SAC_FACTORY in printed and EXTRACTOR in printed

    registry.run_cli(["sac", "--dry-run", *common, *modules])
    constructor_kwargs = json.loads(capsys.readouterr().out)["algorithm"]["constructor_kwargs"]
    assert constructor_kwargs["actor_factory"] == SAC_FACTORY
    assert constructor_kwargs["policy_kwargs"] == {"actor_extractor_class": EXTRACTOR}


# --- checkpoint round trip through the evaluation rebuild ---------------------------------------

ROUND_TRIPS = {
    "ppo-actor_factory": (
        "ppo", ["--actor-factory", PPO_FACTORY, "--num-minibatches", "2"], "actor_factory", PPO_FACTORY,
    ),
    "ppo-actor_extractor": (
        "ppo", ["--actor-extractor", EXTRACTOR, "--num-minibatches", "2"], "actor_extractor", EXTRACTOR,
    ),
    "sac-actor_factory": (
        "sac", ["--actor-factory", SAC_FACTORY, "--buffer-device", "cpu", "--buffer-size", "16"],
        "actor_factory", SAC_FACTORY,
    ),
    "sac-actor_extractor": (
        "sac", ["--actor-extractor", EXTRACTOR, "--buffer-device", "cpu", "--buffer-size", "16"],
        "actor_extractor", EXTRACTOR,
    ),
}


def _env(algorithm: str):
    return _FixedHoldVecEnv(emit_decisions=False) if algorithm == "ppo" else TinyVecEnv()


def _module_of(agent, key: str) -> nn.Module:
    return agent.policy.actor if key == "actor_factory" else agent.policy.actor_extractor


@pytest.mark.parametrize("with_config_json", [True, False], ids=["config_json", "metadata_only"])
@pytest.mark.parametrize("case", sorted(ROUND_TRIPS))
def test_the_evaluation_rebuild_imports_the_recorded_modules(byo, tmp_path, case, with_config_json):
    """The checkpoint records the import path; evaluation's overlay (the
    run's config.json, then the checkpoint metadata) hands it back to the
    builder, which rebuilds a policy the checkpoint loads into strictly."""
    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.common.effective_config import json_value
    from rl_garden.training import evaluation
    from rl_garden.training.online import registry

    algorithm, cli, key, spec = ROUND_TRIPS[case]
    expected_cls = byo.StateBackboneExtractor if key == "actor_extractor" else (
        byo.TanhGaussianPPOActor if algorithm == "ppo" else byo.SquashedSACActor
    )
    builder = evaluation._registered_builder("online", algorithm)
    args = registry.parse_args([algorithm, *cli, "--log-type", "none"])
    assert getattr(args, key) == spec
    agent = builder(args, _env(algorithm), None, None, None)
    assert type(_module_of(agent, key)) is expected_cls

    run_dir = tmp_path / "run"
    path = agent.save(run_dir / "checkpoints" / "checkpoint.pt")
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    hyperparameters = checkpoint["metadata"]["hyperparameters"]
    assert hyperparameters[key] == spec
    assert {"actor_factory", "actor_extractor", "critic_extractor"} & set(hyperparameters) == {key}

    if with_config_json:
        (run_dir / "config.json").write_text(json.dumps({"inputs": json_value(args)}))
        eval_args = evaluation.EvalCheckpointArgs(checkpoint_path=str(path))
    else:
        eval_args = evaluation.EvalCheckpointArgs(
            checkpoint_path=str(path), config_path=str(tmp_path / "missing.json")
        )
    restored_args = evaluation._load_training_args("online", algorithm, eval_args, checkpoint)
    assert getattr(restored_args, key) == spec

    restored = builder(restored_args, _env(algorithm), None, None, None)
    assert type(_module_of(restored, key)) is expected_cls
    restored.load(path, load_replay_buffer=False)
    expected = agent.policy.state_dict()
    actual = restored.policy.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def test_a_default_build_records_no_policy_module(tmp_path):
    from rl_garden.algorithms import SAC
    from rl_garden.common.checkpoint import load_checkpoint_file

    path = SAC(env=TinyVecEnv(), **COMMON_KWARGS).save(tmp_path / "sac.pt")
    hyperparameters = load_checkpoint_file(path, map_location="cpu")["metadata"]["hyperparameters"]
    assert not {"actor_factory", "actor_extractor", "critic_extractor"} & set(hyperparameters)


def _unnamed_sac_factory():
    from rl_garden.networks import SquashedGaussianActor

    return lambda features_dim, action_space: SquashedGaussianActor(
        features_dim, action_space, [8]
    )


UNREBUILDABLE = "actor cannot be rebuilt because its actor_factory had no import path.*module level"


def test_an_unnamed_factory_is_recorded_as_unrebuildable(tmp_path, caplog):
    """A lambda factory is recorded as a sentinel (with a warning); the
    evaluation rebuild restores it onto the Args and the builder's
    ``policy_module_kwargs`` refuses it with a ``ConfigError``."""
    import logging

    from rl_garden.algorithms import SAC
    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.common.cli_args import policy_module_kwargs
    from rl_garden.common.effective_config import ConfigError
    from rl_garden.training import evaluation

    agent = SAC(env=TinyVecEnv(), actor_factory=_unnamed_sac_factory(), **COMMON_KWARGS)
    with pytest.warns(UserWarning, match="actor_factory has no import path"):
        path = agent.save(tmp_path / "sac.pt")
    checkpoint = load_checkpoint_file(path, map_location="cpu")
    recorded = checkpoint["metadata"]["hyperparameters"]["actor_factory"]
    assert recorded.startswith("<unrecorded: <function ") and recorded.endswith(">")

    eval_args = evaluation.EvalCheckpointArgs(
        checkpoint_path=str(path), config_path=str(tmp_path / "missing.json")
    )
    with caplog.at_level(logging.INFO, logger=evaluation.__name__):
        restored_args = evaluation._load_training_args("online", "sac", eval_args, checkpoint)
    assert restored_args.actor_factory == recorded
    assert f"importing actor_factory {recorded} from checkpoint metadata" in caplog.text
    with pytest.raises(ConfigError, match=UNREBUILDABLE):
        policy_module_kwargs(restored_args)


def test_the_evaluation_rebuild_logs_a_restored_import_path(tmp_path, caplog):
    import logging

    from rl_garden.training import evaluation

    checkpoint = {"metadata": {"hyperparameters": {"actor_factory": SAC_FACTORY}}}
    eval_args = evaluation.EvalCheckpointArgs(
        checkpoint_path=str(tmp_path / "sac.pt"), config_path=str(tmp_path / "missing.json")
    )
    with caplog.at_level(logging.INFO, logger=evaluation.__name__):
        restored_args = evaluation._load_training_args("online", "sac", eval_args, checkpoint)
    assert restored_args.actor_factory == SAC_FACTORY
    assert f"importing actor_factory {SAC_FACTORY} from checkpoint metadata" in caplog.text


def test_jsrl_refuses_a_guide_whose_factory_had_no_import_path(tmp_path):
    from rl_garden.algorithms import jsrl
    from rl_garden.common.effective_config import ConfigError

    with pytest.warns(UserWarning, match="actor_factory has no import path"):
        path, agent = _make_calql_checkpoint(tmp_path, actor_factory=_unnamed_sac_factory())
    with pytest.raises(ConfigError, match=UNREBUILDABLE):
        jsrl._load_guide_policy(
            path,
            "calql",
            agent.env.single_observation_space,
            agent.env.single_action_space,
            device=torch.device("cpu"),
        )
