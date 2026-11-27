"""CPU-only contract between CLI fields, registered builders and constructors.

Builders run with default Args and a spaces-only env; construct_agent records
keywords and returns a mock, so no algorithm, replay, simulator or dataset is
created. Only file-dependent preprocessing is stubbed, with its Args reads
covered statically. Runner/shared reads are found from the entry's run function
and its called Python helpers (including lazy imports), counting args attributes
and literal getattr/hasattr names, never arbitrary string constants. Static
reads cover non-default branches; constructor forwarding is checked separately
against the actual captured call, following **kwargs through the MRO.
An additional probe sets the optional module/sharing/group fields so conditional
forwarding counts. Backend config readers are called for every selectable
backend, without creating environments, to cover their computed field names.
The FQL family's warmup_steps -> initial_training_phase mapping is also checked
explicitly, since its names differ and a runner read alone cannot prove wiring.

This test checks that a field is read and that a same-named constructor keyword
is passed. It does not prove unchanged value propagation or downstream consumption.

Exemptions are per check: reading a field does not excuse failing to forward a
same-named constructor parameter. Each exemption must still be necessary.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import pytest
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("online", "offline", "off2on")
ENTRY_EXEMPTIONS = {
    ("offline", "pwm"): "builder reads the multitask manifest to size its task buffer",
}
# check -> (phase, name) -> field -> reason; each check is independent.
# Not applicable: the constructor chain has no same-named parameter.
NOT_APPLICABLE = {'forward': {},
 'read': {('off2on', 'acfql'): {'actor_dropout_rate': 'inherited Args field; this flow actor has '
                                                      'no dropout parameter',
                                'critic_subsample_size': 'inherited Args field; this algorithm '
                                                         'does not subsample its critic ensemble',
                                'std_parameterization': 'inherited Args field; this actor is not a '
                                                        'Gaussian actor'},
          ('off2on', 'awac'): {'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                        'not subsample its critic ensemble'},
          ('off2on', 'fac'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                    'dropout parameter',
                              'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                       'not subsample its critic ensemble',
                              'std_parameterization': 'inherited Args field; this actor is not a '
                                                      'Gaussian actor'},
          ('off2on', 'fino'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                        'not subsample its critic ensemble',
                               'std_parameterization': 'inherited Args field; this actor is not a '
                                                       'Gaussian actor'},
          ('off2on', 'floq'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                        'not subsample its critic ensemble',
                               'std_parameterization': 'inherited Args field; this actor is not a '
                                                       'Gaussian actor'},
          ('off2on', 'fql'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                    'dropout parameter',
                              'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                       'not subsample its critic ensemble',
                              'std_parameterization': 'inherited Args field; this actor is not a '
                                                      'Gaussian actor'},
          ('off2on', 'spot'): {'critic_subsample_size': 'inherited Args field; this algorithm does '
                                                        'not subsample its critic ensemble',
                               'std_parameterization': 'inherited Args field; this actor is not a '
                                                       'Gaussian actor'},
          ('off2on', 'value_flows'): {'actor_dropout_rate': 'inherited Args field; this flow actor '
                                                            'has no dropout parameter',
                                      'critic_subsample_size': 'inherited Args field; this '
                                                               'algorithm does not subsample its '
                                                               'critic ensemble',
                                      'std_parameterization': 'inherited Args field; this actor is '
                                                              'not a Gaussian actor'},
          ('offline', 'a2a_bc'): {'eval_freq': 'inherited logging field; this bespoke runner '
                                               'disables periodic evaluation',
                                  'kernel_init': 'inherited Args field; A2A BC exposes separate '
                                                 'decoder_kernel_init and flow_kernel_init',
                                  'num_eval_steps': 'inherited logging field; this bespoke runner '
                                                    'disables periodic evaluation',
                                  'save_replay_buffer': 'inherited checkpoint field; this runner '
                                                        'has no replay checkpoint support'},
          ('offline', 'awac'): {'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'bcq'): {'utd': 'inherited offline Args field; offline training counts '
                                      'gradient steps directly'},
          ('offline', 'bppo'): {'lr_decay_steps': 'inherited optimizer Args field; this '
                                                  'multi-stage algorithm has no LR scheduler '
                                                  'parameters',
                                'lr_min_ratio': 'inherited optimizer Args field; this multi-stage '
                                                'algorithm has no LR scheduler parameters',
                                'lr_schedule': 'inherited optimizer Args field; this multi-stage '
                                               'algorithm has no LR scheduler parameters',
                                'lr_warmup_steps': 'inherited optimizer Args field; this '
                                                   'multi-stage algorithm has no LR scheduler '
                                                   'parameters',
                                'offline_sampling': 'inherited offline Args field; this '
                                                    'multi-stage algorithm owns its sampling',
                                'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'consistency_distill_bc'): {'eval_freq': 'inherited logging field; this '
                                                               'bespoke runner disables periodic '
                                                               'evaluation',
                                                  'num_eval_steps': 'inherited logging field; this '
                                                                    'bespoke runner disables '
                                                                    'periodic evaluation',
                                                  'save_replay_buffer': 'inherited checkpoint '
                                                                        'field; this runner has no '
                                                                        'replay checkpoint '
                                                                        'support'},
          ('offline', 'diffusion_bc'): {'eval_freq': 'inherited logging field; this bespoke runner '
                                                     'disables periodic evaluation',
                                        'num_eval_steps': 'inherited logging field; this bespoke '
                                                          'runner disables periodic evaluation',
                                        'save_replay_buffer': 'inherited checkpoint field; this '
                                                              'runner has no replay checkpoint '
                                                              'support'},
          ('offline', 'fac'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'utd': 'inherited offline Args field; offline training counts '
                                      'gradient steps directly'},
          ('offline', 'fino'): {'actor_dropout_rate': 'inherited Args field; this flow actor has '
                                                      'no dropout parameter',
                                'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'floq'): {'actor_dropout_rate': 'inherited Args field; this flow actor has '
                                                      'no dropout parameter',
                                'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'fql'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'utd': 'inherited offline Args field; offline training counts '
                                      'gradient steps directly'},
          ('offline', 'hilp'): {'eval_freq': 'inherited logging field; this bespoke runner '
                                             'disables periodic evaluation',
                                'num_eval_steps': 'inherited logging field; this bespoke runner '
                                                  'disables periodic evaluation',
                                'save_replay_buffer': 'inherited checkpoint field; this runner has '
                                                      'no replay checkpoint support'},
          ('offline', 'idql'): {'critic_dropout_rate': 'inherited Args field; this algorithm has '
                                                       'no same-named normalization/dropout '
                                                       'parameter',
                                'critic_use_group_norm': 'inherited Args field; this algorithm has '
                                                         'no same-named normalization/dropout '
                                                         'parameter',
                                'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly',
                                'value_dropout_rate': 'inherited Args field; this algorithm has no '
                                                      'same-named normalization/dropout parameter',
                                'value_use_group_norm': 'inherited Args field; this algorithm has '
                                                        'no same-named normalization/dropout '
                                                        'parameter'},
          ('offline', 'opal'): {'eval_freq': 'inherited logging field; this bespoke runner '
                                             'disables periodic evaluation',
                                'num_eval_steps': 'inherited logging field; this bespoke runner '
                                                  'disables periodic evaluation',
                                'save_replay_buffer': 'inherited checkpoint field; this runner has '
                                                      'no replay checkpoint support'},
          ('offline', 'plas'): {'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'qam'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'actor_use_group_norm': 'inherited Args field; this algorithm has '
                                                       'no same-named normalization/dropout '
                                                       'parameter',
                               'critic_dropout_rate': 'inherited Args field; this algorithm has no '
                                                      'same-named normalization/dropout parameter',
                               'critic_use_group_norm': 'inherited Args field; this algorithm has '
                                                        'no same-named normalization/dropout '
                                                        'parameter',
                               'num_groups': 'inherited Args field; this algorithm has no '
                                             'same-named normalization/dropout parameter',
                               'utd': 'inherited offline Args field; offline training counts '
                                      'gradient steps directly'},
          ('offline', 'qgf'): {'actor_dropout_rate': 'inherited Args field; this flow actor has no '
                                                     'dropout parameter',
                               'actor_use_group_norm': 'inherited Args field; this algorithm has '
                                                       'no same-named normalization/dropout '
                                                       'parameter',
                               'critic_dropout_rate': 'inherited Args field; this algorithm has no '
                                                      'same-named normalization/dropout parameter',
                               'critic_use_group_norm': 'inherited Args field; this algorithm has '
                                                        'no same-named normalization/dropout '
                                                        'parameter',
                               'num_groups': 'inherited Args field; this algorithm has no '
                                             'same-named normalization/dropout parameter',
                               'utd': 'inherited offline Args field; offline training counts '
                                      'gradient steps directly'},
          ('offline', 'rebrac'): {'utd': 'inherited offline Args field; offline training counts '
                                         'gradient steps directly'},
          ('offline', 'rssm_pretrain'): {'eval_freq': 'inherited logging field; this bespoke '
                                                      'runner disables periodic evaluation',
                                         'num_eval_steps': 'inherited logging field; this bespoke '
                                                           'runner disables periodic evaluation',
                                         'save_replay_buffer': 'inherited checkpoint field; this '
                                                               'runner has no replay checkpoint '
                                                               'support'},
          ('offline', 'spot'): {'utd': 'inherited offline Args field; offline training counts '
                                       'gradient steps directly'},
          ('offline', 'td3_bc'): {'utd': 'inherited offline Args field; offline training counts '
                                         'gradient steps directly'},
          ('offline', 'tdmpc2_multitask'): {'eval_freq': 'inherited logging field; this bespoke '
                                                         'runner disables periodic evaluation',
                                            'num_eval_steps': 'inherited logging field; this '
                                                              'bespoke runner disables periodic '
                                                              'evaluation'},
          ('offline', 'unio4'): {'lr_decay_steps': 'inherited optimizer Args field; this '
                                                   'multi-stage algorithm has no LR scheduler '
                                                   'parameters',
                                 'lr_min_ratio': 'inherited optimizer Args field; this multi-stage '
                                                 'algorithm has no LR scheduler parameters',
                                 'lr_schedule': 'inherited optimizer Args field; this multi-stage '
                                                'algorithm has no LR scheduler parameters',
                                 'lr_warmup_steps': 'inherited optimizer Args field; this '
                                                    'multi-stage algorithm has no LR scheduler '
                                                    'parameters',
                                 'offline_sampling': 'inherited offline Args field; this '
                                                     'multi-stage algorithm owns its sampling',
                                 'utd': 'inherited offline Args field; offline training counts '
                                        'gradient steps directly'},
          ('offline', 'unio4_ope'): {'lr_decay_steps': 'inherited optimizer Args field; this '
                                                       'multi-stage algorithm has no LR scheduler '
                                                       'parameters',
                                     'lr_min_ratio': 'inherited optimizer Args field; this '
                                                     'multi-stage algorithm has no LR scheduler '
                                                     'parameters',
                                     'lr_schedule': 'inherited optimizer Args field; this '
                                                    'multi-stage algorithm has no LR scheduler '
                                                    'parameters',
                                     'lr_warmup_steps': 'inherited optimizer Args field; this '
                                                        'multi-stage algorithm has no LR scheduler '
                                                        'parameters',
                                     'offline_sampling': 'inherited offline Args field; this '
                                                         'multi-stage algorithm owns its sampling',
                                     'utd': 'inherited offline Args field; offline training counts '
                                            'gradient steps directly'},
          ('offline', 'value_flows'): {'actor_dropout_rate': 'inherited Args field; this flow '
                                                             'actor has no dropout parameter',
                                       'utd': 'inherited offline Args field; offline training '
                                              'counts gradient steps directly'},
          ('online', 'diffusion_cm_distill_online'): {'save_replay_buffer': 'inherited checkpoint '
                                                                            'field; this runner '
                                                                            'has no replay '
                                                                            'checkpoint support'},
          ('online', 'dppo'): {'save_replay_buffer': 'inherited checkpoint field; this runner has '
                                                     'no replay checkpoint support'},
          ('online', 'flow_ppo'): {'save_replay_buffer': 'inherited checkpoint field; this runner '
                                                         'has no replay checkpoint support'},
          ('online', 'gail'): {'save_replay_buffer': 'inherited checkpoint field; this runner has '
                                                     'no replay checkpoint support'},
          ('online', 'policy_distillation'): {'save_replay_buffer': 'inherited checkpoint field; '
                                                                    'this runner has no replay '
                                                                    'checkpoint support'},
          ('online', 'ppo'): {'save_replay_buffer': 'inherited checkpoint field; this runner has '
                                                    'no replay checkpoint support'},
          ('online', 'recurrent_ppo'): {'save_replay_buffer': 'inherited checkpoint field; this '
                                                              'runner has no replay checkpoint '
                                                              'support'},
          ('online', 'supe'): {'dataset_backend': 'SUPE skill relabeling loads H5 windows '
                                                  'directly; backend selection is unsupported',
                               'success_key': 'SUPE skill relabeling does not read a success '
                                              'label'},
          ('online', 'transformer_ppo'): {'save_replay_buffer': 'inherited checkpoint field; this '
                                                                'runner has no replay checkpoint '
                                                                'support'}}}

# Ignored today: accepted by the constructor, not passed by the builder.
IGNORED_TODAY = {'forward': {('offline', 'bppo'): {'encoder_sharing': 'accepted by the constructor, not passed by '
                                                      'the builder',
                                   'obs_groups': 'accepted by the constructor, not passed by the '
                                                 'builder'},
             ('offline', 'unio4'): {'encoder_sharing': 'accepted by the constructor, not passed by '
                                                       'the builder',
                                    'obs_groups': 'accepted by the constructor, not passed by the '
                                                  'builder'},
             ('offline', 'unio4_ope'): {'encoder_sharing': 'accepted by the constructor, not '
                                                           'passed by the builder',
                                        'obs_groups': 'accepted by the constructor, not passed by '
                                                      'the builder'},
             ('online', 'acrlpd'): {'nstep': 'the builder rejects any value but 1: the replay '
                                             'window is horizon_length'},
             ('online', 'dagger'): {'eval_freq': 'accepted by the constructor, not passed by the '
                                                 'builder',
                                    'num_eval_steps': 'accepted by the constructor, not passed by '
                                                      'the builder'},
             ('online', 'recurrent_sac'): {'critic_backbone_type': 'accepted by the constructor, '
                                                                   'not passed by the builder',
                                           'encoder_sharing': 'accepted by the constructor, not '
                                                              'passed by the builder',
                                           'obs_groups': 'accepted by the constructor, not passed '
                                                         'by the builder'},
             ('online', 'sac_flow'): {'actor_log_std_min': 'accepted by the constructor, not '
                                                           'passed by the builder',
                                      'actor_log_std_mode': 'accepted by the constructor, not '
                                                            'passed by the builder',
                                      'actor_use_layer_norm': 'accepted by the constructor, not '
                                                              'passed by the builder',
                                      'encoder_sharing': 'accepted by the constructor, not passed '
                                                         'by the builder'},
             ('online', 'transformer_sac'): {'critic_backbone_type': 'accepted by the constructor, '
                                                                     'not passed by the builder',
                                             'encoder_sharing': 'accepted by the constructor, not '
                                                                'passed by the builder',
                                             'obs_groups': 'accepted by the constructor, not '
                                                           'passed by the builder'}},
 'read': {('online', 'recurrent_sac'): {'critic_backbone_type': 'accepted by the constructor, not '
                                                                'passed by the builder'},
          ('online', 'sac_flow'): {'actor_log_std_min': 'accepted by the constructor, not passed '
                                                        'by the builder',
                                   'actor_log_std_mode': 'accepted by the constructor, not passed '
                                                         'by the builder',
                                   'actor_use_layer_norm': 'accepted by the constructor, not '
                                                           'passed by the builder'},
          ('online', 'transformer_sac'): {'critic_backbone_type': 'accepted by the constructor, '
                                                                  'not passed by the builder'}}}

def registered_entries():
    entries = {}
    for phase in PHASES:
        registry = importlib.import_module(f"rl_garden.training.{phase}.registry").registry
        registry.discover()
        entries.update({(phase, name): entry for name, entry in registry.entries().items()})
    return entries


@lru_cache(None)
def _module_tree(module):
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.is_file():
        return None
    return ast.parse(path.read_text())


def _static_reads(module, function, seen=None):
    """Follow called module functions, not every builder in an imported module."""
    seen = set() if seen is None else seen
    key = (module, function)
    if key in seen or not module.startswith("rl_garden."):
        return set()
    seen.add(key)
    tree = _module_tree(module)
    if tree is None:
        return set()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    # Registry normalization/validation are methods, called before every runner.
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            functions.update({f"{node.name}.{fn.name}": fn for fn in node.body
                              if isinstance(fn, ast.FunctionDef)})
    fn = functions.get(function)
    if fn is None:
        return set()
    imports = {}
    for node in [*tree.body, *ast.walk(fn)]:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module, alias.name)
    reads = set()
    args_names = {"args", "rank_args", "normalized_args", "train_args"}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id in args_names):
            reads.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if (name in {"getattr", "hasattr"} and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id in args_names
                    and isinstance(node.args[1], ast.Constant)):
                reads.add(node.args[1].value)
            target = imports.get(name, (module, name))
            reads.update(_static_reads(*target, seen))
        # Function-valued callbacks such as make_env_request also receive Args.
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and (node.id in imports or node.id in functions):
            reads.update(_static_reads(*imports.get(node.id, (module, node.id)), seen))
    return reads


def constructor_parameters(cls):
    parameters = {}
    for base in cls.__mro__:
        init = vars(base).get("__init__")
        if init is None or base is object:
            continue
        signature = inspect.signature(init)
        parameters.update({name: parameter for name, parameter in signature.parameters.items()
                           if name != "self" and parameter.kind not in
                           (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
                           and name not in parameters})
        if not any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            break
    return parameters


class FakeEnv(gym.Env):
    """The spaces-only portion of test_entrypoint_builds' fake env contract."""
    metadata = {"autoreset_mode": None}
    num_envs = 2
    device = "cpu"
    single_observation_space = spaces.Dict({"state": spaces.Box(-1.0, 1.0, (5,))})
    single_action_space = spaces.Box(-1.0, 1.0, (3,))
    observation_space = single_observation_space
    action_space = spaces.Box(-1.0, 1.0, (2, 3))


def capture_builder(phase, name, entry, *, conditional=False, args=None):
    args = entry.args_cls() if args is None else args
    if conditional:
        if (phase, name) == ("online", "sac_flow"):
            args.obs = dataclasses.replace(args.obs, rgb=("front",))
        if hasattr(args, "encoder_sharing"):
            args.encoder_sharing = "shared_critic_grad"
        if (hasattr(args, "obs_groups")
                and "obs_groups" in constructor_parameters(entry.algorithm_cls())):
            from rl_garden.observations import ObsGroups
            args.obs_groups = ObsGroups(actor=("state",), critic=("state",))
        if hasattr(args, "actor_factory"):
            args.actor_factory = "torch.nn:Linear"
    if name == "policy_distillation":
        args.teacher_checkpoint = "unused.pt"
        args.teacher_obs_keys = ["state"]
        args.student_obs_keys = ["state"]
    fields = {field.name for field in dataclasses.fields(args)}
    reads = set()

    def record_read(self, field):
        if field in fields:
            reads.add(field)
        return object.__getattribute__(self, field)

    args.__class__ = type(type(args).__name__, (type(args),), {"__getattribute__": record_read})
    captured = {}

    def construct(target, **kwargs):
        assert not captured, f"{phase}/{name}: builder constructed more than one agent"
        captured.update(target=target, kwargs=kwargs)
        return MagicMock()

    # These helpers otherwise load external inputs before construct_agent.
    with (patch("torch.cuda.is_available", return_value=False),
          patch("rl_garden.training.inspection.construct_agent", construct),
          patch("rl_garden.algorithms.supe.load_opal_vae",
                return_value=SimpleNamespace(skill_dim=4, decoder=None)),
          patch("rl_garden.envs.wrappers.SkillActionWrapper", side_effect=lambda env, *a, **k: env),
          patch("rl_garden.buffers.tdmpc2_multitask_dataset.infer_multitask_dataset_specs",
                return_value=(["t0"], [5], [3], [10])),
          patch("rl_garden.training.online.entrypoints.policy_distillation._build_teacher_policy",
                return_value=None)):
        env = FakeEnv()
        if phase == "offline":
            from rl_garden.algorithms.offline import OfflineEnvSpec
            spec = OfflineEnvSpec(env.single_observation_space, env.single_action_space, 1)
            entry.build_agent(args, spec, None, None)
        else:
            entry.build_agent(args, env, FakeEnv(), None, None)
    # Backend selection reads config fields by computed names; exercise only
    # config_from_args, never backend environment factories.
    if hasattr(args, "resolve_backend_config"):
        from rl_garden.common.env_args import EnvBackendArgs
        from rl_garden.envs.backend_registry import resolve_backend_config
        for field in dataclasses.fields(EnvBackendArgs):
            if field.name != "env_backend":
                resolve_backend_config(field.name, args)
    assert captured, f"{phase}/{name}: builder did not call construct_agent"
    return fields, reads, captured


def drift_failures(phase, name):
    entry = registered_entries()[phase, name]
    if (phase, name) in ENTRY_EXEMPTIONS:
        return []
    fields, builder_reads, captured = capture_builder(phase, name, entry)
    reads = builder_reads | _static_reads(entry.run_fn.__module__, entry.run_fn.__name__)
    # Shared preflight receives every entry's Args before run_fn.
    for function in ("_normalize_runtime", "_validate_config"):
        reads |= _static_reads("rl_garden.training.algorithm_registry",
                               f"BaseAlgorithmRegistry.{function}")
    builder = entry.build_agent
    reads |= _static_reads(builder.__module__, builder.__name__)
    _, conditional_reads, conditional_capture = capture_builder(phase, name, entry, conditional=True)
    reads |= conditional_reads
    passed = captured["kwargs"].keys() | conditional_capture["kwargs"].keys()
    missing = {
        "read": fields - reads,
        "forward": (fields & constructor_parameters(captured["target"]).keys())
                   - passed,
    }
    if phase == "off2on" and name in {"fql", "acfql", "fac", "fino", "floq", "value_flows"}:
        if "initial_training_phase" not in passed:
            missing["forward"].add("warmup_steps")
    failures = []
    for check, missing_fields in missing.items():
        exemptions = {
            **NOT_APPLICABLE[check].get((phase, name), {}),
            **IGNORED_TODAY[check].get((phase, name), {}),
        }
        for field in sorted(missing_fields - exemptions.keys()):
            failures.append(f"{phase}/{name}: {field}: {check} check failed")
        for field in sorted(exemptions.keys() - missing_fields):
            failures.append(f"{phase}/{name}: {field}: stale {check} exemption")
    return failures


ENTRIES = registered_entries()


@pytest.mark.parametrize("phase,name", sorted(ENTRIES), ids=[f"{p}/{n}" for p, n in sorted(ENTRIES)])
def test_args_builder_drift(phase, name):
    assert not (failures := drift_failures(phase, name)), "\n".join(failures)


def test_exemptions_name_registered_entries():
    assert ENTRY_EXEMPTIONS.keys() <= ENTRIES.keys()
    for exemptions in (NOT_APPLICABLE, IGNORED_TODAY):
        for table in exemptions.values():
            assert table.keys() <= ENTRIES.keys()
            assert not (table.keys() & ENTRY_EXEMPTIONS.keys())
            assert all(reason.strip() for fields in table.values() for reason in fields.values())
    for check in NOT_APPLICABLE:
        for key, fields in NOT_APPLICABLE[check].items():
            parameters = constructor_parameters(ENTRIES[key].algorithm_cls())
            assert not (fields.keys() & parameters.keys())
        for key, fields in IGNORED_TODAY[check].items():
            parameters = constructor_parameters(ENTRIES[key].algorithm_cls())
            assert fields.keys() <= parameters.keys()
            assert not (fields.keys() & NOT_APPLICABLE[check].get(key, {}).keys())


FLOW_ENTRIES = ("fql", "acfql", "fac", "fino", "floq", "value_flows")
VALUE_PROBES = [
    ("online", name, field, value)
    for name in ("explore", "supe")
    for field, value in (("target_policy_noise", 0.17), ("target_noise_clip", 0.31),
                         ("std_parameterization", "uniform"), ("actor_q_agg", "mean"))
] + [
    ("online", "acrlpd", field, value)
    for field, value in (("q_landscape_diagnostics", True), ("q_landscape_num_actions", 19),
                         ("q_landscape_batch_size", 13), ("q_mc_diagnostics", True),
                         ("critic_backbone_type", "mlp_resnet"))
] + [
    ("offline", "wsrl", field, value)
    for field, value in (("use_sarsa_reference", True), ("sarsa_lr", 0.0017),
                         ("sarsa_hidden_dims", (17, 23)))
] + [("off2on", "awac", "std_parameterization", "uniform")] + [
    ("off2on", name, field, value)
    for name in FLOW_ENTRIES
    for field, value in (("actor_use_group_norm", True), ("critic_use_group_norm", True),
                         ("num_groups", 16))
] + [("online", "sac_flow", "critic_backbone_type", "mlp_resnet")]


@pytest.mark.parametrize("phase,name,field,value", VALUE_PROBES)
def test_added_builder_keywords_preserve_nondefault_values(phase, name, field, value):
    entry = ENTRIES[phase, name]
    args = entry.args_cls()
    assert getattr(args, field) != value
    setattr(args, field, value)
    _, _, captured = capture_builder(phase, name, entry, args=args)
    assert captured["kwargs"][field] == value


@pytest.mark.parametrize("name", FLOW_ENTRIES)
@pytest.mark.parametrize("warmup_steps", [0, 7])
def test_flow_builder_warmup_phase_values(name, warmup_steps):
    entry = ENTRIES["off2on", name]
    args = entry.args_cls(warmup_steps=warmup_steps)
    _, _, captured = capture_builder("off2on", name, entry, args=args)
    phase = captured["kwargs"]["initial_training_phase"]
    if warmup_steps == 0:
        assert phase is None
    else:
        assert phase.duration_steps == warmup_steps
        assert phase.update_actor is False
        assert phase.update_critic is False
        assert phase.update_encoder is False


@pytest.mark.parametrize("name", ["explore", "supe"])
def test_policy_module_flags_reach_builder(name):
    from torch import nn

    entry = ENTRIES["online", name]
    args = entry.args_cls(actor_factory="torch.nn:Linear", actor_extractor="torch.nn:Identity",
                          critic_extractor="torch.nn:Flatten")
    _, _, captured = capture_builder("online", name, entry, args=args)
    assert captured["kwargs"]["actor_factory"] is nn.Linear
    assert captured["kwargs"]["policy_kwargs"] == {
        "actor_extractor_class": nn.Identity, "critic_extractor_class": nn.Flatten,
    }
