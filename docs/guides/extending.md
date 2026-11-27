# Extending rl-garden from Your Own Package

A downstream package extends rl-garden by **subclassing and registration**: it
subclasses an rl-garden algorithm (and, where needed, a policy or actor),
registers the subclass on a phase registry, and then trains, logs,
checkpoints and evaluates it with rl-garden's own CLI, runner, rollout and
evaluation code, without editing or copying rl-garden.

This guide is the extension contract: the registration API, the builders and
the protected methods listed here are what a downstream package may rely on.
A hook named here counts as a consumer under the parent-hook rule in
[`AGENTS.md`](https://github.com/JaimeParker/rl-garden/blob/main/AGENTS.md); a change to one of them updates this guide in
the same commit.

The reference implementation is the fixture package
[`tests/fixtures/downstream_extension/`](https://github.com/JaimeParker/rl-garden/tree/main/tests/fixtures/downstream_extension/)
(`rlg_downstream_example`), exercised end to end by
`tests/test_downstream_extension.py`. It registers `aux_split_rppo`, a
`RecurrentPPO` whose policy has a two-head action mean, with an auxiliary loss
on a per-step env info and an extra metric, and two extra CLI flags; the test
trains it, saves a checkpoint and evaluates it with `--algorithm auto`. The examples below are
taken from it.

For PPO ablations that only change the network between the observation and the
heads, use a [policy graph](policy-graphs.md) (YAML, no Python subclass). A
task-specific network, loss or rollout detail goes through this contract.

For actor factories and external feature extractors without an algorithm subclass,
see {doc}`/tutorials/08-bring-your-own-policy`.

## 1. Registering an algorithm

### The registry call

Each training phase has a registry: `rl_garden.training.online.registry`,
`...offline.registry`, `...off2on.registry` (import `registry` from the
package, e.g. `from rl_garden.training.online.registry import registry`).

```python
registry.register(
    name,              # the CLI subcommand
    args_cls,          # its own Args dataclass (a type may be registered once)
    run_fn,            # run_fn(args): trains
    algorithm_cls=...,  # zero-arg callable returning the algorithm class
    build_agent=...,    # the builder run_fn constructs its agent with
    training_only_inputs=...,  # Args fields only training needs (evaluation)
)
```

- `algorithm_cls` is a thunk (import inside it) so that registration never
  imports `rl_garden.algorithms`. Preflight reads class attributes through it
  (`encoder_sharing`, `encoder_sharing_choices`, `stateful_policy`,
  `configured_sequence_stage`), and evaluation with `--algorithm auto` matches a
  checkpoint's class name against it.
- `build_agent` is `build_agent(args, env, eval_env, logger, checkpoint_dir)`
  for the online and off2on phases and `build_agent(args, env_spec, logger,
  eval_env)` for the offline phase. Evaluation rebuilds the agent with it.
- `training_only_inputs` names the Args fields that point at something only
  training needs: a dataset, a warm-start, teacher or demonstration source, a
  disk-backed replay directory. Evaluation sets them to their absent value
  before it calls `build_agent`, so the builder must build an agent without
  them (a run that requires one checks it in `run_fn` or, for a fresh start,
  in the builder). A tuple of names uses each field's default, which must be
  `None` or `""`; a mapping gives the value itself, e.g.
  `{"demo_env_id": None, "online_ratio": 1.0}`. An unknown field raises
  `ValueError`. Without the argument every field reaches the builder as the
  run had it.
- A duplicate name or a reused Args type raises `ValueError`. Dispatch matches
  the exact Args type, so a subclassed Args needs its own registration.

### Getting the registering module imported

The `rlgarden.algorithms` entry-point group: declare the module that calls
`registry.register(...)` in your package's `pyproject.toml`,

```toml
[project.entry-points."rlgarden.algorithms"]
my_package = "my_package.registration"
```

Every phase registry imports its in-tree entrypoints and then every module of
this group, once per process, before it parses a command. The module may
register on any phase. A module that fails to import raises a `RuntimeError`
naming its entry point; an external name that clashes with an in-tree one
fails in its `register` call. The same pattern exists for env backends
(`rlgarden.env_backends`, see [Environment backends](#7-environment-backends)) and policy-graph nodes (`rlgarden.graph_nodes`).

A package that is not installed imports its registering module from its own
launcher before calling `registry.run_cli()`.

### Commands

Installing rl-garden provides `rl-garden-train-online`,
`rl-garden-train-offline`, `rl-garden-train-off2on` and `rl-garden-eval`
(`rl_garden/training/cli.py`); the `examples/train_online.py`,
`examples/pretrain_offline.py`, `examples/train_off2on.py` and
`examples/eval_checkpoint.py` scripts of a source checkout are the same
commands. With the entry point installed, `rl-garden-train-online my_algo ...`
finds your algorithm; `--help`, `--print-config`, `--dry-run`, `--config`
presets and `config.json` all cover its extra fields.

## 2. Args and builders

Subclass the Args of the algorithm you extend and add fields; they become CLI
flags:

```python
from dataclasses import dataclass

from rl_garden.training.online.entrypoints.recurrent_ppo import (
    RecurrentPPOArgs,
    build_recurrent_ppo,
    run_recurrent_ppo,
)


@dataclass
class AuxSplitRecurrentPPOArgs(RecurrentPPOArgs):
    aux_weight: float = 0.1
    aux_hidden: int = 16


def build_aux_split_rppo(args, env, eval_env, logger, checkpoint_dir):
    from my_package.algorithm import AuxSplitRecurrentPPO

    return build_recurrent_ppo(
        args, env, eval_env, logger, checkpoint_dir,
        algorithm_cls=AuxSplitRecurrentPPO,
        aux_weight=args.aux_weight,
        aux_hidden=args.aux_hidden,
    )


def run_aux_split_rppo(args):
    run_recurrent_ppo(args, build_agent=build_aux_split_rppo)
```

The in-tree builders `build_ppo`, `build_recurrent_ppo`,
`build_transformer_ppo`, `build_gail`, `build_sac`, `build_rlpd`,
`build_rlpd_hybrid`, `build_recurrent_sac` and `build_transformer_sac` take
keyword-only `algorithm_cls` (the in-tree class when omitted; `build_sac`
picks `SACDDP` under multi-GPU DDP only then) and forward any other keyword to
the constructor; `build_rlpd` and `build_rlpd_hybrid` merge those keywords
after the ones they read from `args`, so one of them replaces the value taken
from `args` (for example `use_grasp_penalty` or `memory_efficient_buffer`).
`RLPDHybrid`'s `num_grippers` (1 by default) reaches it this way: the last
`num_grippers` action dims are the grippers, in order, and its DQN head picks
one of `3 ** num_grippers` joint gripper actions, labelled and decoded by
`encode_gripper_labels` / `decode_gripper_index` in
`rl_garden.policies.rlpd_hybrid_policy`.
The `run_*` functions of the other builders take `build_agent=` (the module's
builder when omitted); `run_rlpd` and `run_rlpd_hybrid` do not.

A builder exposing external policy modules mixes `PolicyModuleArgs` into its Args
and forwards `policy_module_kwargs(args)` from `rl_garden.common.cli_args`.
It resolves `actor_factory` and the actor/critic extractor classes; omit unset
values so the standard construction path remains active.

A builder written from scratch maps Args to constructor kwargs with the public
helpers `ppo_common_kwargs` / `ppo_observation_kwargs`
(`rl_garden/training/online/entrypoints/ppo.py`) or `sac_common_kwargs` /
`sac_observation_kwargs` (`.../sac.py`), and must construct the agent through
`rl_garden.training.inspection.construct_agent(cls, **kwargs)`: it records the
call for `--dry-run` and the materialized `config.json`, and a run whose builder
bypassed it fails.

## 3. The algorithm subclass

### Hyperparameter dataclasses

`AWACCore`, `TD3BCCore`, `IQLCore`, `FQLCore`, `QAMCore`, `QGFCore`,
`BCQCore`, `PLASCore`, `IDQL` and their concrete classes use a Params dataclass in
`rl_garden.hyperparameters`, importable without `rl_garden.algorithms`, as
the single definition of each hyperparameter's name, type, default and
description. Params subclasses follow the core hierarchy, re-declaring only
defaults they override; shared groups come from
`rl_garden.hyperparameters.shared`. A shared parent contains only fields all
its consumers use.

The FQL family separates its shared `FQLParams` from scalar-critic options
(`FQLCriticParams`), the distillation coefficient (`FQLDistillationParams`)
and the published FQL recipe switch (`FQLReferenceParams`, used by `FQL`
and `Off2OnFQL`). `FACParams` inherits the scalar-critic settings without
FQL's unused `alpha`; `ValueFlowsParams` inherits the distillation settings without the
scalar-critic options. Existing compatibility keywords stay
plain constructor arguments. `horizon_length` remains an action-chunking
mixin argument. The rollout Args compose orchestration separately so sibling
actor/critic defaults cannot decide learning settings.

`QAMParams`, `QGFParams`, `BCQParams`, `PLASParams` and `IDQLParams`
are independent dataclasses. Only QGF, BCQ, PLAS and IDQL inherit the shared
optimization mixin, whose defaults they use in full. QAM owns its optimization
fields because its clipping default is `1.0`; its Args compose the offline run
mixins without the shared optimization defaults. QAM and QGF keep
`horizon_length` as a plain action-chunking argument. BCQ and PLAS preserve
explicit architecture conversion in their builders; their leaf Args keep the
CLI architecture default `(400, 300)` while Params default to `None`.
IDQL's leaf Args retain its larger CLI critic ensemble. Existing unused CLI
flags stay in Args and are not exposed as Params fields.

`BCParams`, `FlowBCParams`, `MeanFlowBCParams` and `A2ABCParams` own their
family settings. `DiffusionBCParams` and `ConsistencyDistillBCParams` inherit
the common diffusion and network settings from `DiffusionPolicyParams`,
keeping teacher optimization/EMA and student consistency settings separate.
Only BC, FlowBC and MeanFlowBC use the shared optimization mixin in full.
`DAggerParams` inherits `BCParams`: DAgger adds collection rounds and a demo
buffer, with no additional policy-update settings. Its constructor delegates
Params resolution and binding to BC while keeping those loop inputs and the
injected expert as keywords. Dataset and teacher paths, network factories,
network kwargs and observation wiring remain constructor keywords.
DiffusionBC resolves `ema_start_step=None` from dataset size on the live
attribute; the Params snapshot retains None. A2ABC keeps its legacy
`activation_fn` CLI mapping when `cnn_activation_fn` is None; any non-None
`cnn_activation_fn` takes precedence, including the constructor default `relu`.
The CLI cannot select a None CNN activation because the older flag requires
a named activation. Direct constructor calls still accept None.

`BPPOParams` and `UniO4Params` share `BPPOCriticParams`, the settings
consumed by their shared V/Q critic and optimizers. `UniO4OPEParams` extends
`UniO4Params` with dynamics training and OPE gating. Hidden-dimension settings
in these classes are active architecture parameters. The legacy actor CLI
names (`actor_use_layer_norm`, `actor_use_group_norm`, `actor_dropout_rate`)
remain supported: a canonical `use_layer_norm`, `use_group_norm` or
`dropout_rate` value takes precedence whenever it is not None, including
values equal to the Params default. None uses the legacy actor flag.
The canonical normalization flags take `True`, `False` or `None` values;
the legacy flags retain their switch syntax. No dropout is selected through
`actor_dropout_rate=None` with `dropout_rate=None`.
`UniO4OPE._bind_params` calls the base binding method and resolves the live
dynamics widths and weight-decay sequences to lists before model setup;
its Params snapshot keeps the input sequences.

`HILPParams` and `OPALParams` are independent settings for their standalone
dataset learners. Dataset paths and trajectory limits remain constructor
keywords. OPAL metadata keeps the names and resolved values SUPE uses to
rebuild its frozen VAE. `DDPGParams` uses the shared optimization settings;
`TD3Params` extends it with policy delay and target smoothing. Online
`drqv2` and `td3` expose the constructor's learning-rate schedule fields.
Replay/mmap options and injected `policy_kwargs` remain constructor keywords.
The DDPG rollout, replay and actor/target extension hooks retain their contracts.

`PPOParams` owns the Gaussian PPO update and head settings. `SequencePPO`
uses that same schema; `RecurrentPPOParams`, `TransformerPPOParams` and
`GAILParams` extend it with their latent-stage or discriminator settings.
`ChainPPOCoreParams` owns the shared chain actor/critic update settings;
`DPPOParams` and `FlowPPOParams` add their respective sampling objectives,
and `DiffusionCMDistillOnlineParams` extends DPPO with consistency updates.
`PolicyDistillationParams` owns the independent teacher-student objective.
The on-policy loop arguments (including decision-mask mode and critic step
selection), teacher/checkpoint inputs, observation routing,
policy graphs and deprecated architecture aliases remain constructor keywords.

Forwarding children separate parameter overrides from parent operational
keywords, resolve the child's schema and pass the resolved object to their
parent. `GAIL._bind_params` and `DiffusionCMDistillOnline._bind_params` call
the base binding method and preserve their tuple/list architecture attributes
before model setup. Resolved architectures and reward horizons remain live
attributes. Optimizer learning-rate schedules, discriminator state and
consistency target updates retain their existing checkpoint hooks; they do
not modify the Params snapshot.

`ppo_common_kwargs` accepts the keyword-only `params_class` argument,
defaulting to `PPOParams`. The PPO-family builders pass their selected
algorithm's schema so subclass parameters reach the constructor through the
same projection. Policy distillation retains its explicit architecture list
conversion and its leaf Args default `(256, 256)`.

`DreamerV3Params` and `NCRLParams` are siblings under `DreamerCoreParams`:
NCRL takes its RSSM architecture from its pretrained checkpoint and exposes its
own optimizers. `RSSMArchitectureParams` shares constructor-owned architecture
settings between DreamerV3 and RSSMPretrain; `RSSMLossParams` shares their
world-model loss and sequence settings with NCRL. The existing run-level
arguments remain constructor keywords.

`DreamerV3.dreamer_inputs_class` defaults to `None`, selecting the concrete
`params_class` for resolution and binding. NCRL sets it to `DreamerV3Params`
and binds its own snapshot before calling the parent. The parent then copies
its resolved fields without replacing that snapshot. This preserves checkpoint
architecture and the `actor_ent` → `act_entropy` and `discount` →
`discount_horizon` translations before model setup. Subclasses extend their
own Params schema through `params_class`.

The model-based `TDMPC2Params` and `TDMPC2MultitaskParams` inherit
`TDMPC2CoreParams`. `PWMParams` and `SuperMimicParams` own their policy
learning settings. The concrete constructors resolve and bind the snapshot
before calling their core parameter helpers with `params=...`; the helpers
validate inputs and resolve derived attributes. Model-based host classes and
step mixins retain their loop contracts. PWM reads world-model architecture
from its checkpoint; those settings are not policy-learning Params fields.
Mixed precision, replay layout, injected world-model objects and loop-owned
arguments remain constructor keywords. Live optimizer schedules and return
normalizers do not mutate the snapshot.

The online SAC subtree uses `SACParams`, with subclasses for flow actors,
sequence encoders, RLPD, hybrid grippers, ExPLORe and ACRLPD. `SACDDP`
inherits SAC's schema; SUPE inherits ExPLORe's schema and keeps its OPAL
checkpoint and injected VAE as inputs. `SACCore` remains independent of
Params, including its offline and FlashSAC consumers. Inherited settings
that a policy ignores remain accepted for compatibility.

SAC binds before resolving `alpha_lr`, `net_arch` and entropy settings.
Metadata keeps the input entropy target and the resolved live target under
separate keys. Flow, ExPLORe and hybrid binding overrides preserve converted
list/tuple/integer attributes. ACRLPD derives the entropy target passed to
SAC while retaining the original input snapshot. Forwarding children resolve
their own schema, so caller overrides take precedence over child defaults.
`sac_common_kwargs` accepts a keyword-only `params_class`; each entry passes
its own schema by name, independently of an injected algorithm class.

`FlashSACParams` independently owns FlashSAC's distributional-critic and
learning settings. `OfflineSACParams` owns the offline SAC settings and shares
only the identical optimizer fields with `OfflineOptimizationArgs`.
`EDACParams` extends it with `eta` and the ten-critic default. EDAC resolves
its concrete schema and passes the object to OfflineSAC, preserving caller
overrides. OfflineSAC keeps the input entropy target separate from its live
resolved target; FlashSAC metadata records the resolved target. Compilation,
AMP, asymmetric observation wiring and loop settings remain plain keywords.

`CQLParams` owns the shared conservative loss settings; `CalQLParams` adds
calibration and reference-value settings. `WSRLParams` adds the online objective
switch, and `Off2OnCalQLParams` retains conservatism online by default.
Constructors resolve their concrete schema; `CQLCore._init_cql_params` binds it
once. `CalQLCore._init_calql_params` sets the fields needed before parent setup,
and its `_bind_params` override preserves tuple-valued SARSA widths. CQL setup
keeps `use_calql=False` until the calibrated constructor restores the requested
value after parent setup. Stage switches and checkpoint restoration mutate live
objective values, while the Params snapshot retains constructor inputs.
`_cql_kwargs` accepts a keyword-only `params_class`; each offline builder names
its schema. It keeps the forced automatic entropy settings and uses legacy
width flags only when `net_arch` is `None`. Encoder wiring, compilation,
replay sampling and stage/evaluation orchestration remain plain keywords.

The constructor accepts `params=` and keyword overrides, calls
`resolve_params(self.params_class, params, overrides)` before `super().__init__`
so resolved values can reach parents, and keeps its validation. Loop-level,
operational and observation arguments remain plain keywords. The core calls
`self._bind_params(params)` before resolving derived values such as
`net_arch`. `self.params` is a snapshot of constructor inputs; values that
change afterwards remain live instance attributes, and metadata reads those
live values. Metadata overrides after `super()._checkpoint_metadata()` still
win, including the resolved architecture and encoder settings.

The entry's Args inherits the Params class. A CLI default that differs from
the class default is re-declared on the entry's leaf Args. A field's type
belongs to Params and admits the values supported by the underlying code;
leaves do not override a field solely to change its type. Deprecated constructor
aliases remain plain keywords, without Args flags or metadata fields. The builder passes
`params_kwargs(args, cls.params_class)` as constructor keyword arguments
through `construct_agent`. Add a family row to
`tests/test_hyperparameter_contract.py`; `tests/test_args_default_shadowing.py`
and `tests/test_args_builder_drift.py` also guard defaults and builder wiring.

| Seam | Signature | What the base does | Consumers |
|---|---|---|---|
| `BaseAlgorithm.params_class` | `Optional[type]` | `None` for algorithms with constructor keywords; opt in with the hyperparameter dataclass | `AWACCore`, `TD3BCCore`, `IQLCore`, `FQLCore`, `QAMCore`, `QGFCore`, `BCQCore`, `PLASCore`, `IDQL`, `BC`, `DAgger`, `FlowBC`, `MeanFlowBC`, `DiffusionBC`, `A2ABC`, `ConsistencyDistillBC`, `BPPO`, `UniO4`, `UniO4OPE`, `HILP`, `OPAL`, `DDPG`, `TD3`, `PPO`, `SequencePPO`, `RecurrentPPO`, `TransformerPPO`, `GAIL`, `ChainPPOCore`, `DPPO`, `FlowPPO`, `DiffusionCMDistillOnline`, `PolicyDistillation`, `TDMPC2`, `TDMPC2Multitask`, `PWM`, `SuperMimic`, `SAC`, `SACDDP`, `SACFlow`, `SequenceSAC`, `RecurrentSAC`, `TransformerSAC`, `RLPD`, `RLPDHybrid`, `ExPLORe`, `SUPE`, `ACRLPD`, `DreamerV3`, `NCRL`, `RSSMPretrain`, `CQLCore`, `CalQLCore`, `CQL`, `CalQL`, `WSRL`, `Off2OnCalQL` and concrete classes (declared on concrete classes) |
| `BaseAlgorithm._bind_params` | `(params: Any) -> None` | stores the resolved object as `self.params` and copies each field of `self.params_class` onto the instance; call only with an opted-in Params class; no validation or derived values | `AWACCore`, `TD3BCCore`, `IQLCore`, `FQLCore`, `QAMCore`, `QGFCore`, `BCQCore`, `PLASCore`, `IDQL`, `BC`, `DAgger`, `FlowBC`, `MeanFlowBC`, `DiffusionBC`, `A2ABC`, `ConsistencyDistillBC`, `BPPO`, `UniO4`, `UniO4OPE`, `HILP`, `OPAL`, `DDPG`, `TD3`, `PPO`, `SequencePPO`, `RecurrentPPO`, `TransformerPPO`, `GAIL`, `ChainPPOCore`, `DPPO`, `FlowPPO`, `DiffusionCMDistillOnline`, `PolicyDistillation`, `TDMPC2`, `TDMPC2Multitask`, `PWM`, `SuperMimic`, `SAC`, `SACDDP`, `SACFlow`, `SequenceSAC`, `RecurrentSAC`, `TransformerSAC`, `RLPD`, `RLPDHybrid`, `ExPLORe`, `SUPE`, `ACRLPD`, `DreamerV3`, `NCRL`, `RSSMPretrain`, `FlashSAC`, `OfflineSAC`, `EDAC`, `CQLCore`, `CalQLCore`, `CQL`, `CalQL`, `WSRL`, `Off2OnCalQL` and concrete classes |
| `BaseAlgorithm._checkpoint_metadata` | `() -> dict[str, Any]` | seed and device, plus each Params field's live instance value when `params_class` is set; unchanged seed/device mapping when it is `None` | `AWACCore`, `TD3BCCore`, `IQLCore`, `FQLCore`, `QAMCore`, `QGFCore`, `BCQCore`, `PLASCore`, `IDQL`, `BC`, `DAgger`, `FlowBC`, `MeanFlowBC`, `DiffusionBC`, `A2ABC`, `ConsistencyDistillBC`, `BPPO`, `UniO4`, `UniO4OPE`, `HILP`, `OPAL`, `DDPG`, `TD3`, `PPO`, `SequencePPO`, `RecurrentPPO`, `TransformerPPO`, `GAIL`, `ChainPPOCore`, `DPPO`, `FlowPPO`, `DiffusionCMDistillOnline`, `PolicyDistillation`, `TDMPC2`, `TDMPC2Multitask`, `PWM`, `SuperMimic`, `SAC`, `SACDDP`, `SACFlow`, `SequenceSAC`, `RecurrentSAC`, `TransformerSAC`, `RLPD`, `RLPDHybrid`, `ExPLORe`, `SUPE`, `ACRLPD`, `DreamerV3`, `NCRL`, `RSSMPretrain`, `FlashSAC`, `OfflineSAC`, `EDAC`, `CQLCore`, `CalQLCore`, `CQL`, `CalQL`, `WSRL`, `Off2OnCalQL` and concrete classes (extend with cooperative `super()`) |

### Constructor ordering

`PPO.__init__` and `SAC.__init__` end by calling `self._setup_model()`, which
builds the policy. Set additional attributes your overrides read **before**
`super().__init__()`. Params fields, including `RecurrentPPO`'s `rnn_*` fields,
are bound by PPO before model setup:

```python
def __init__(self, env: Any, *, aux_weight: float = 0.1, aux_hidden: int = 16, **kwargs: Any):
    if kwargs.get("policy_graph") is not None:
        raise ConfigError(f"{type(self).__name__} builds its own policy; drop --policy-graph.")
    # Set before super().__init__(): it ends by building the policy,
    # which reads them.
    self.aux_weight = aux_weight
    self.aux_hidden = aux_hidden
    self._aux_logit_means: list[torch.Tensor] = []
    super().__init__(env, **kwargs)
```

Forward the remaining parent arguments through `**kwargs` as above. PPO
and SAC resolve their declared Params fields and reject unknown keywords. A subclass whose policy replaces
the actor rejects `--policy-graph` as above: a policy graph builds a
`GraphPPOPolicy`, not `policy_class`.

### Network: `policy_class`, `_policy_head_kwargs` and `_build_actor`

`PPO._build_policy(observation_space)` builds `policy_class` (a `PPOPolicy`
subclass; `SequencePPO` needs a `RecurrentPPOPolicy` subclass) with the
extractors and the keywords of `_policy_head_kwargs(observation_space)`;
`_setup_model` then moves the policy to the device and builds the optimizer
over `policy.parameters()`, so anything attached to the policy is trained,
gradient-clipped, all-reduced under DDP and checkpointed. Set the class, and
pass your own constructor arguments through `_policy_head_kwargs`:

```python
class AuxSplitRecurrentPPO(RecurrentPPO):
    policy_class = AuxSplitRecurrentPolicy
    action_split = 2

    def _policy_head_kwargs(self, observation_space):
        return {
            **super()._policy_head_kwargs(observation_space),
            "action_split": self.action_split,
            "aux_hidden": self.aux_hidden,
        }
```

The policy builds its modules in its constructor. `PPOPolicy._build_actor(
features_dim, action_space, hidden_dims, **actor_options)` is the seam for the
actor: it is called once, after the extractors and before the value head, with
the width of the features the actor reads (for a recurrent or transformer
policy the sequence latent's width, not the extractor's), the resolved `pi`
widths and the algorithm's actor options. The default builds `DiagGaussianActor`,
or calls `actor_factory(features_dim, action_space)` when the policy was given
one (the options are not forwarded to it). An override builds its own actor and
does not call `actor_factory`. The module it returns must satisfy
`rl_garden.policies.contracts.PPOActor` (`action_log_prob`,
`evaluate_action_log_prob`, `deterministic_action`, `clamp_action`; shapes in
`rl_garden/policies/contracts.py`): right after
`_build_actor`, `PPOPolicy._actor_capabilities(features_dim, action_space)`
validates it (`validate_ppo_actor`: a `TypeError`/`ValueError` at construction)
and its result is stored as `policy.actor_capabilities`. Modules the policy adds
after `super().__init__()` (here `aux_head`) are policy submodules:

```python
class AuxSplitRecurrentPolicy(RecurrentPPOPolicy):
    #: The update's evaluate calls return ``{"latent": <the actor's features>}``.
    evaluation_outputs = True

    def __init__(self, *args: Any, action_split: int, aux_hidden: int, **kwargs: Any) -> None:
        # Read by _build_actor, which super().__init__() calls.
        self.action_split = action_split
        super().__init__(*args, **kwargs)
        # A policy submodule, so the optimizer, the gradient clip and the
        # checkpoint all cover it.
        width = self.recurrent_encoder.features_dim
        self.aux_head = nn.Sequential(nn.Linear(width, aux_hidden), nn.ReLU(), nn.Linear(aux_hidden, 1))

    def _build_actor(
        self,
        features_dim: int,
        action_space: spaces.Box,
        hidden_dims: Any,
        **actor_options: Any,
    ) -> nn.Module:
        # features_dim is the recurrent latent's width, which the actor reads.
        act_dim = int(np.prod(action_space.shape))
        return SharedInputSplitActor(
            [features_dim, features_dim],
            action_space,
            [(0, self.action_split), (self.action_split, act_dim)],
            [hidden_dims, []],
            **actor_options,
        )
```

`SplitDiagGaussianActor` expects one features tensor per head, while
`PPOPolicy` and `RecurrentPPOPolicy` pass one tensor. A three-line `_mean`
override hands it to every head:

```python
class SharedInputSplitActor(SplitDiagGaussianActor):
    def _mean(self, features: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            features = [features] * len(self.heads)
        return super()._mean(features)
```

Any `PPOActor` works on the plain path; its entropy may be `None`, which drops
the entropy bonus and the `entropy` metric; a positive `ent_coef` is rejected at construction. `PPO._check_actor_capabilities()`
(run by `_setup_model` right after the policy is built) rejects, with a
`ValueError`, `--lr-schedule adaptive_kl` unless `actor_capabilities.is_gaussian`
(a `log_std` tensor and a `forward` returning a `Normal`) and `--decision-level`
unless `actor_capabilities.supports_per_dim_log_prob` (`sum_dims=False`);
decision-level `policy/log_std/<j>` is logged only for a Gaussian actor.
`actor_factory` cannot be combined with a policy graph. A policy graph (`--policy-graph`) builds a
`GraphPPOPolicy` and ignores `policy_class`; its `_policy_head_kwargs` reach
`GraphPPOPolicy` too, so keywords only your `policy_class` accepts belong to a
subclass that rejects `--policy-graph`.

A package that swaps the actor after construction instead
(`policy.actor = ...` following `super()._build_policy(...)`) must size it by
the sequence latent in a recurrent or transformer policy
(`policy.recurrent_encoder.features_dim`), the replaced actor has already
consumed random numbers, so initial weights differ from a policy that builds
the new actor directly, and it skips the contract validation and
`actor_capabilities`. Prefer `_build_actor` (or `actor_factory`); this holds for
`SACPolicy` as well.

A SAC-family policy is swapped by setting `policy_class` (and extending
`_policy_kwargs` for its own keywords, see the SAC family seam table); its actor
by `SACPolicy._build_actor` or `actor_factory`. A non-Gaussian SAC actor
implements `SACActor.diagnostics(features)` (or inherits `SACActorBase`'s
default), which `SACPolicy.actor_diagnostics` delegates to;
`SACCore._compute_actor_diagnostics` stays the algorithm-side seam for batches
the policy cannot read flat (`SequenceSAC` returns `{}`).

(extending-ppo-seams)=
### PPO seam table

The protected PPO methods and attributes a subclass overrides, at this version:

| Seam | Purpose |
|---|---|
| `PPO.policy_class` | the policy `_build_policy` builds when no policy graph is given |
| `PPO._policy_head_kwargs(observation_space)` | the head keywords every built policy receives (`net_arch`, actor init options, norm/dropout options, backbone types); extend with `{**super()._policy_head_kwargs(observation_space), ...}`; read once per `_build_policy` call |
| `PPO._build_policy(observation_space)` | the policy itself; prefer `policy_class` and `_policy_head_kwargs` to overriding it |
| `PPO(actor_factory=...)` / `PPOPolicy(actor_factory=...)` | `fn(features_dim, action_space) -> PPOActor` that the default `_build_actor` calls; reaches the policy through `_policy_head_kwargs` only when set; excludes `policy_graph` |
| `PPOPolicy._build_actor(features_dim, action_space, hidden_dims, **actor_options)` | the actor module; see above |
| `PPOPolicy._actor_capabilities(features_dim, action_space) -> PPOActorCapabilities` | checks the built actor, right after `_build_actor`; default `validate_ppo_actor`; `GraphPPOPolicy` overrides it for its multi-sink actor. Stored as `policy.actor_capabilities` |
| `PPO._check_actor_capabilities()` | construction-time checks of `policy.actor_capabilities` against the run's options (`adaptive_kl`, `decision_level`); extend with `super()` |
| `PPOPolicy.rollout_extra_specs()` | per-step extras the rollout buffer stores, `{key: (shape, dtype)}`; each key is read from `infos[key]` on every env step (a missing key raises) and arrives in `data.extras[key]`. Return `{**super().rollout_extra_specs(), ...}` |
| `PPOPolicy.evaluation_outputs` (class attribute) | when `True`, the update calls the policy's evaluate method with `return_outputs=True` and the trailing dict lands in `MinibatchEval.outputs`: `{"latent": <the features the actor read>}` for `PPOPolicy` and `RecurrentPPOPolicy` (rows in minibatch order, T-major for sequences), the graph outputs for `GraphPPOPolicy` |
| `PPO.rollout_buffer_class` (class attribute) | the buffer `_build_rollout_buffer()` allocates once in `_setup_model`, with `RolloutBuffer`'s constructor keywords; `RolloutBuffer` (`RecurrentRolloutBuffer` for `SequencePPO`). A subclass changes GAE or storage in a buffer subclass |
| `PPO._rollout_extra_specs()` | the extras the buffer declares, `{key: (shape, dtype)}`, read once when it is built; default `policy.rollout_extra_specs()`. Extend with `{**super()._rollout_extra_specs(), key: (shape, dtype)}` for a tensor the algorithm computes itself |
| `PPO._rollout_extras()` | per-step tensors the algorithm computed during the current rollout step, `{key: (num_envs, *shape)}` (e.g. stashed on `self` in `_rollout_policy` or `_after_env_step`); default `{}`. Every key must be declared by `_rollout_extra_specs()` and returned on every step (both raise `ValueError` otherwise); a key returned here is not read from `infos`. Arrives in `data.extras[key]` |
| `OnPolicyAlgorithm._after_env_step(rewards, terminations, truncations, infos)` | called once per rollout step right after `env.step()`, before the episode cut, the bootstrap value and `rollout_buffer.add`; default no-op. The tensors are the ones stored next |
| `OnPolicyAlgorithm._next_done(terminations, truncations, infos)` | the episode cut, `(num_envs,)` on the device: stored as the step's `dones` (GAE stops there), the next step's `episode_starts` (a recurrent policy resets there) and the window's last done; default `terminations \| truncations` |
| `OnPolicyAlgorithm._bootstrap_mask(terminations, truncations)` / `_compute_final_values(infos, done_mask, hidden)` | which ended episodes bootstrap their GAE target from `final_observation` (default per `bootstrap_at_done`), and that value (0 elsewhere); independent of `_next_done` |
| `PPO._aux_loss(data, evaluation, decision_mask)` | the auxiliary loss of a minibatch: `(loss or None, {name: detached scalar})`; the loss is added to the PPO loss before its single `backward()`, each term is averaged over the update and returned by `train()` as `aux/<name>` (logged as `losses/aux/<name>`). Extend `super()`'s result so policy-graph losses still apply. Reduce a per-row loss with `rl_garden.algorithms.ppo.masked_mean(row_loss, decision_mask, decision_mask.sum())` |
| `PPO._policy_loss(advantages=, ratio=, clip_coef=, mask=)` | the scalar policy loss from the (normalized) advantages and the ratio, 1-D over the rows; `mask` is the decision mask or `None`; default the clipped surrogate |
| `PPO._value_loss(values=, old_values=, returns=, clip_coef=, decision_mask=)` | the scalar value loss (`values` carries gradients); default `ppo_terms.clipped_value_loss` (`0.5 * mse`, clipped when `clip_vloss`), over every row or the decision rows (`per_step_critic`) under a decision mask |
| `PPO._entropy_term(terms)` | the scalar entropy bonus, or `None`; default `terms.row_mean(terms.entropy)`, the mean over the policy-loss rows (`PolicyTerms`: `advantages`, `logratio`, `ratio`, `entropy`, `row_mean`, `decisions`), `None` when the actor reports no entropy |
| `PPO._total_loss(policy_loss=, value_loss=, entropy_loss=, aux_loss=)` | the loss of the minibatch's single `backward()`; default `policy_loss - ent_coef * entropy_loss + vf_coef * value_loss (+ aux_loss)`, without the entropy term when `entropy_loss` is `None`, reported as `loss` |
| `PPO._should_stop_early(approx_kl)` | whether `target_kl` stops the update at this minibatch, before its gradient step; default: the approx-KL averaged over DDP ranks (`allreduce_mean`) exceeds `target_kl` (never when it is `None`), so every rank stops at the same minibatch. An override must decide identically on every rank |
| `PPO._extra_update_metrics(stats)` | extra `train()` metrics, merged last and logged as `losses/<key>`; `stats` is the update's `UpdateStats` (per-minibatch losses, KLs, clip fractions, decision counts, aux terms, `stopped_early`); default `{}` |
| `PPO.train()` | one update; `metrics = super().train()`, then add keys (prefer `_extra_update_metrics`) |
| `BaseAlgorithm._log_update_metrics(metrics, step)` | where `train()`'s keys are logged; call `super()` so the base keys stay logged |
| `_checkpoint_metadata()` | extra hyperparameters; see section 4 |

### Where the PPO update can be changed

One override per scenario, the rest of the loop unchanged:

- per-row policy-loss formula (dual clip, a ratio penalty): `_policy_loss`;
- value or entropy term: `_value_loss` / `_entropy_term`; how the terms
  combine: `_total_loss`;
- bootstrap value of an ended episode: `_bootstrap_mask` /
  `_compute_final_values`; where an episode is cut for GAE and the
  recurrent state: `_next_done`;
- a per-step quantity computed during the rollout and read in the update:
  `_rollout_extras` with `_rollout_extra_specs` (stash it in
  `_rollout_policy` or `_after_env_step`, read `data.extras` in
  `_aux_loss`);
- a metric reduced over the update: `_extra_update_metrics`.

`train()` and `_ppo_minibatch_update(...)` are built from further named steps
(`_run_epochs`, `_minibatch_step`, `_policy_terms`, `_kl_diagnostics`,
`_adaptive_kl_step`, `_optimizer_step`, `_update_metrics`) that keep the
collectives in a fixed order on every DDP rank; they are not part of this
contract.

### Chain PPO (`DPPO`, `FlowPPO`)

`DPPO` and `FlowPPO` (and `DiffusionCMDistillOnline`, a `DPPO`) train a
denoising or flow chain with PPO's clip ratio applied per chain step. They are
not `PPO` subclasses: both subclass `ChainPPOCore(OnPolicyAlgorithm)`
(`rl_garden/algorithms/chain_ppo_core.py`). The core owns the rollout step and
the stored chain, the advantages (normalized and quantile-clamped over the
population the class attribute `advantage_normalization` names: `"minibatch"` for
each minibatch's pairs, as in `DPPO`, or `"rollout"` for the rollout's env steps,
as in `FlowPPO` and by default), the pair-minibatch loop, the two-optimizer step with critic warmup
and the actor gradient clip over the actor optimizer's parameters, `target_kl`
checked after each step, the LR schedulers (checkpoint key
`lr_scheduler_states`), the averaged metrics and explained variance. A subclass
builds `self.policy` in `_setup_model` and then calls
`_build_chain_training_state(...)`, and implements `_sample_rollout_chain(obs)`
and `_chain_policy_terms(minibatch) -> ChainPolicyTerms` (its log-ratio, ratio,
advantages and clip coefficient). The loss helpers they share with `PPO`
(`ppo_clip_policy_loss`, `clipped_value_loss`, `approx_kl_clipfrac`,
`explained_variance`, `encoder_checkpoint_metadata`) live in
`rl_garden/algorithms/ppo_terms.py`.

`policy_class` and `_policy_head_kwargs` are the algorithm side of the policy
seams, `_build_actor`, `rollout_extra_specs` and `evaluation_outputs` the
policy side. The reference package declares its auxiliary target in the
policy:

```python
def rollout_extra_specs(self) -> ExtraSpecs:
    return {**super().rollout_extra_specs(), AUX_TARGET_KEY: ((1,), torch.float32)}
```

and implements the auxiliary loss on the latent the policy hands back:

```python
def _aux_loss(self, data, evaluation, decision_mask):
    aux_loss, terms = super()._aux_loss(data, evaluation, decision_mask)
    logits = self.policy.aux_head(evaluation.outputs["latent"])
    target = data.extras[AUX_TARGET_KEY].reshape(-1, 1).to(logits.dtype)
    row_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")[:, 0]
    if decision_mask is None:
        bce = row_loss.mean()
    else:
        bce = masked_mean(row_loss, decision_mask, decision_mask.sum())
    self._aux_logit_means.append(logits.detach().mean())
    weighted = self.aux_weight * bce
    aux_loss = weighted if aux_loss is None else aux_loss + weighted
    # train() returns the terms as aux/<name>; they are logged as
    # losses/aux/<name>.
    return aux_loss, {**terms, "bce": bce.detach()}
```

Under DDP every rank must return a loss reaching the same parameters on every
minibatch. A custom evaluate method of your policy returns the outputs dict
last when called with `return_outputs=True`.

(extending-sac-seams)=
### SAC family seam table

`SAC` (and `RLPD`, `SequenceSAC`, `SACFlow`, which set their own
`policy_class`) builds the policy, then `q_optimizer` and `actor_optimizer`,
then the temperature optimizer and the LR schedulers. `SACCore` owns the
update: `train` runs, per gradient step, `_critic_step`, then, on actor steps,
`_actor_step`, `_alpha_step` and `_post_actor_update`; `train_high_utd` (taken
when `utd` is an integer above 1) runs `_critic_step` on every minibatch slice
and then one actor step on the full batch. The seams, at this version:

| Seam | Default | What a subclass may assume |
|---|---|---|
| `SAC.policy_class` (class attribute) | `SACPolicy` (`RLPDPolicy`, `RLPDHybridPolicy`, `RecurrentSACPolicy`, `SACFlowPolicy` in those subclasses; `SACFlowPolicy` only overrides `_build_actor`) | built once by `_build_policy` as `policy_class(**_policy_kwargs(observation_space))`, then moved to the device |
| `SAC._policy_kwargs(observation_space) -> dict` | `observation_space`, `action_space`, the critic/actor options, `actor_factory` (only when given), then the extractors (`actor_extractor`, `critic_extractor`, `encoder_sharing`) | building it builds the extractors, so call it (and `super()`) once; extend with `{**super()._policy_kwargs(observation_space), ...}` |
| `CQLCore.policy_class` (class attribute) / `CQLCore._policy_kwargs(observation_space) -> dict` | `SACPolicy`; `observation_space`, `action_space`, CQL's critic/actor options, `actor_factory` (only when given), then the extractors | `_setup_model` builds `policy_class(**_policy_kwargs(observation_space))` once and moves it to the device (`CQL`, `CalQL`, `WSRL` and their rollout shells); building the keywords builds the extractors, so call it (and `super()`) once; extend with `{**super()._policy_kwargs(observation_space), ...}` |
| `OfflineSAC.policy_class` (class attribute) / `OfflineSAC._policy_kwargs(observation_space) -> dict` | `SACPolicy`; `observation_space`, `action_space`, `net_arch`, `n_critics`, `critic_subsample_size`, `actor_factory` (only when given), then the extractors | as `CQLCore`'s (`EDAC`) |
| `actor_factory` (constructor keyword of `SAC`, `OfflineSAC`, `CQL`, `CalQL`, `WSRL` and their off2on shells; class attribute on `SACCore`, `None`) | `fn(features_dim, action_space) -> SACActor` that the default `SACPolicy._build_actor` calls; recorded in the checkpoint as an import path | `SACFlow` and `SequenceSAC` reject it (their policies build their own actor) |
| `SACPolicy._build_actor(features_dim, action_space, **actor_options) -> nn.Module` | `SquashedGaussianActor` with the policy's Gaussian options, or `actor_factory(features_dim, action_space)` (options not forwarded) | called once, right after the critic and `critic_target` (the RNG position of the old inline build), with the actor's feature width (`actor_feature_dim`'s compressor output when set); a subclass sets its own options as plain attributes before `super().__init__()`. The result is validated (`validate_sac_actor`) and its flags stored as `policy.actor_capabilities` |
| `SACCore._check_actor_capabilities(capabilities)` | `ValueError` for `ent_coef="auto"` with `stochastic=False`; `CQLCore` adds `cql_importance_sample=True` with `log_prob_is_density=False`, `ACRLPDCore` `bc_alpha > 0` with `supports_evaluate_log_prob=False` | runs once in `_setup_model` right after the policy is built; extend with `super()._check_actor_capabilities(capabilities)` |
| `SACCore._target_policy_action(next_action) -> Tensor` | `next_action` unchanged when `target_policy_noise` is 0 (no random numbers), else plus clipped Gaussian noise (`target_noise_clip`), clamped to the action bounds | the next action the critic target reads (TD3's target smoothing) |
| `SACCore._actor_q_value(critic_features, action) -> Tensor` | the online ensemble reduced by `actor_q_agg` (`"min"`, `"mean"`, `"first"`) | the `(B, 1)` Q-value the actor loss maximizes; `ACRLPD` keeps its reference critic mean and rejects `actor_q_agg != "min"` |
| `SAC._make_optimizers(*, exclude_bias_from_decay=False) -> (q_optimizer, actor_optimizer)` | Adam (AdamW under weight decay) over `policy.critic_and_encoder_parameters()` and `policy.actor_parameters()`; `exclude_bias_from_decay=True` puts `ndim <= 1` parameters in a no-decay group | called once in `_setup_model`, before the temperature optimizer and the schedulers, which are bound to what it returns |
| `SACPolicy.critic_and_encoder_parameters()` / `actor_parameters()` | the critic and the critic's extractor / the actor, plus the actor extractor when the actor loss trains it (`"separate"`, `"shared"`) | the two optimizers hold exactly these; a module the policy adds is trained only once one of them yields it, after `super()`'s parameters (optimizer state is saved by position) |
| `SACCore._critic_features(obs, *, target) -> Tensor` | `policy.extract_critic_features(obs)`, detached as a whole when the training phase freezes the encoder | called once per critic step on its batch (once per high-UTD slice), inside `_critic_loss`; an override that reads the features calls `super()` and returns the same tensor |
| `SACCore._critic_loss(data) -> (loss, info)` | the TD loss; `info` holds detached scalars (`td_loss`, `target_q`, `critic_loss`, `predicted_q`) | `info` keys are averaged over the update and returned by `train()` when `compute_info`; add terms with `loss, info = super()._critic_loss(data)` |
| `SACCore._critic_step(data) -> (loss, info)` | `_critic_loss`, `q_optimizer` zero_grad/backward, `_sync_ddp_grads` and `_clip_grad_norm` over `critic_and_encoder_parameters()`, step, critic scheduler, `_post_critic_update` | `policy.prepare_batch_all` already ran on `data`; wrap `super()._critic_step(data)` |
| `SACCore._actor_step(data) -> (loss, log_prob)` | `_actor_loss_from_batch`, `actor_optimizer` zero_grad/backward, DDP sync and clip over `actor_parameters()`, step, actor scheduler | runs every `policy_frequency` gradient steps; `log_prob` is detached |
| `SACCore._alpha_step(log_prob) -> alpha_loss or None` | one `alpha_optimizer` step on `_alpha_loss`; `None` with a fixed temperature | follows every `_actor_step` |
| `SACCore._post_critic_update(data, critic_info)` | no-op | runs after every critic step, high-UTD slices included |
| `SACCore._post_actor_update(data) -> dict` | `{}` | runs after every actor step and its temperature step; returned detached scalars are logged like `_critic_loss`'s info |
| `BaseAlgorithm._optimizer_names()` | `("q_optimizer", "actor_optimizer", "alpha_optimizer", "cql_alpha_optimizer")` | an extra optimizer is built in `_setup_model` after `super()._setup_model()` and named here (`(*super()._optimizer_names(), "my_optimizer")`) so checkpoints save and load it |

`SequenceSAC` does not build on `SAC._policy_kwargs`: its policy takes a
different keyword set, so a subclass of it extends its own `_policy_kwargs`
(`SACFlow` extends `SAC._policy_kwargs` with its flow keywords). A class that overrides `_build_policy` itself ignores
`policy_class`; no in-tree SAC subclass does.

### Off-policy loop and replay seams

These are stable contract points: the real-world package
(`real_world_rl`, in the downstream `real-world-rl` repository,
which carries rl-garden as a submodule) subclasses `SAC`, `RLPD` and `RLPDHybrid` through
them, so a change to a signature or to when the loop calls a hook updates this
table and `tests/test_off_policy_seams.py` in the same commit. `learn()` starts with
`env.reset(seed=...)`, `replay_buffer.cut_running_episodes()` and
`_on_env_reset`, then runs, per environment step: `_rollout_action`, `env.step`, `_on_env_step`,
`_replay_buffer_add_kwargs`
and `_replay_buffer_step_kwargs` (merged into one `replay_buffer.add(obs,
real_next_obs, replay_action, rewards, stop_bootstrap, **kwargs)` call), then
`_post_rollout_step`.

| Hook | Signature | What the base does | In-tree overriders |
|---|---|---|---|
| `OffPolicyAlgorithm._rollout_action` | `(obs, learning_has_started: bool) -> (replay_action, env_action, action_context)` | `_policy_action(obs)` mixed with `_explore_action(obs)` at `random_action_prob` while an initial training phase is active; otherwise `_explore_action` before `learning_starts` and `_policy_action` after; returns `(actions, actions, None)`. `replay_action` is stored in the buffer, `env_action` goes to `env.step`, `action_context` (a dict or `None`) reaches `_replay_buffer_add_kwargs` and `_post_rollout_step` of the same step | `DDPG`, `FlashSAC`, `SequenceSAC`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `NCRL`, `SuperMimic`, `_SPOTRolloutTrainingShell` |
| `OffPolicyAlgorithm._explore_action` | `(obs) -> Tensor` | uniform in `[-1, 1]`, shaped like `env.action_space` (all envs), on the algorithm's device | `NCRL`, `SuperMimic` |
| `OffPolicyAlgorithm._on_env_step` | `(rewards, terminations, truncations) -> None` | no-op; called right after every training `env.step` with the tensors it returned, before the replay write | `FlashSAC` (updates its reward normalizer) |
| `OffPolicyAlgorithm._replay_buffer_add_kwargs` | `(action_context, obs, next_obs, real_next_obs, infos, need_final_obs) -> dict` | `{}`; the dict is passed as extra keywords to `replay_buffer.add` | `RLPDHybrid` (`grasp_penalty` from `infos`), `SequenceSAC`, `TransformerSAC`, `DreamerV3`, `SuperMimic` |
| `OffPolicyAlgorithm._replay_buffer_step_kwargs` | `(terminations, truncations) -> dict` | `{}` (`SAC`: `{"episode_end": terminations \| truncations}` when `nstep > 1`); merged after the add keywords, so a key present in both takes this value | `SAC`, `DDPG`, `FlashSAC`, `SequenceSAC`, `RLPDHybrid`, `ChunkedRolloutMixin` (`ACRLPD`, `ACFQL`), `_CalQLRolloutTrainingShell`, `ModelBasedAlgorithm` |
| `OffPolicyAlgorithm._on_env_reset` | `(obs) -> None` | no-op; called once by `learn()` after its `env.reset(seed=...)` and the replay buffer's `cut_running_episodes()` (later resets happen inside `env.step`, see `terminations` / `truncations` in `_post_rollout_step`) | `SequenceSAC`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `NCRL`, `SuperMimic` |
| `OffPolicyAlgorithm._post_rollout_step` | `(action_context, terminations, truncations, infos) -> None` | no-op; runs after the replay write and before `obs = next_obs` | `SequenceSAC`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `SuperMimic` |
| `BaseAlgorithm._eval_action` | `(obs) -> Tensor` | `policy.predict(obs, deterministic=True)` under `no_grad`, observations moved to the policy device | `PWM`, `PolicyDistillation`, `SuperMimic`, `OnPolicyAlgorithm` |
| `BaseAlgorithm._eval_action_and_critic_action` | `(obs) -> (env_action, critic_action)` | `(a, a)` with `a = _eval_action(obs)`; the first is executed, the second is what the critic scores | `SequenceSAC`, `DreamerV3`, `NCRL`, `SequencePPO`, `ActionChunkingMixin` (`ChunkedRolloutMixin`, `QAM`, `QGF`: the next action of each env's chunk, and that whole chunk) |
| `BaseAlgorithm._eval_start_hook` | `() -> None` | no-op; called once after `eval_env.reset()`, before the eval loop (`SACCore` resets its Q-vs-MC bookkeeping here) | `SACCore`, `SequenceSAC` (does not call `super()`), `TDMPC2`, `DreamerV3`, `SuperMimic`, `SequencePPO`, `ActionChunkingMixin` |
| `BaseAlgorithm._eval_step_hook` | `(obs_before, critic_action, rewards, terminations, truncations, infos) -> None` | no-op; called after every eval `env.step` (`SACCore` accumulates Q-vs-MC when `q_mc_diagnostics`) | same classes as `_eval_start_hook` |
| `SAC._build_replay_buffer` | `() -> BaseReplayBuffer` | `NStepReplayBuffer` when `nstep > 1`, else `ReplayBuffer` over `num_envs` envs; called once in `_setup_model` | `RecurrentSAC`, `TransformerSAC`, `RLPDHybrid`, `ActionChunkingMixin` (`ACRLPD`, `ACFQL`, `QAM`, `QGF`), `CalQLCore` |
| `BaseReplayBuffer.cut_running_episodes` | `() -> None` | no-op; called once per `learn()` after its `env.reset`, and by any other loop that resets the env itself while it fills a buffer. The episodes still running at the last stored row are truncated there: the stored `dones` stay as they are. A buffer class that computes returns, bootstrap observations or observation windows from several rows (n-step, frame stack, sequence window, Monte Carlo return, next action) must override it so that none of them takes the last row before the cut and the first row after it for consecutive steps of one episode (what each in-tree buffer does, and the limitations: [Checkpoint](checkpoint.md#replay-buffer)); on an empty buffer it changes nothing and it uses no random numbers. Such a class also sets `rows_form_trajectories = True`, so `add_flat_transitions` loads a dataset into it as one contiguous stretch per env. `MemoryEfficientReplayBuffer` overrides the hook but leaves `rows_form_trajectories` `False` (and has no `episode_ends` column or `_episode_end` forwarding), so loading a flat dataset into it is not supported: its frames are striped across the env columns and episode ends are not forwarded, which stacks frames of other episodes; it is meant to be filled one step at a time through `add`. `RehearsalSequenceBuffer` is not a `BaseReplayBuffer`: its `cut_running_episodes` delegates to its `.online` buffer | `NStepSamplingMixin`, `LazyNextNStepReplayBuffer`, `MemoryEfficientReplayBuffer`, `SequenceReplayBuffer`, `RehearsalSequenceBuffer` (delegates to `.online`), `MCReplayBufferMixin`, `ReBRACReplayBuffer` |
| `SAC._policy_action_space` | `() -> spaces.Box` | `env.single_action_space`; read by `_policy_kwargs` and by the buffer builders of the overriders | `ActionChunkingMixin` (`ACRLPD`, `ACFQL`, `QAM`, `QGF`) |
| `SACCore._sample_train_batch` | `(batch_size) -> sample` | `replay_buffer.sample(batch_size)`; called once per gradient step (once per group under high UTD). The sample is a dataclass with `obs`, `next_obs`, `actions`, `rewards`, `dones` and the fields named by `_extra_batch_slice_keys` | `PriorDataReplayMixin` (mixes in the prior-data buffer), `ExPLORe`, `Off2OnReplayMixin`, `ActionChunkingMixin` (flattens after `super()` samples), `QAMCore`, `QGFCore` (sampling-mode dispatch), `OfflineSAC`, `CQL` |
| `SACCore._target_action_log_prob` | `(data) -> (next_action, next_log_prob, next_actor_features)` | `_actor_action_log_prob(data.next_obs)` with `_target_policy_action` applied to its action; log probability and actor features stay unchanged. Downstream subclasses may override the target action construction | none |
| `SACCore._actor_loss_from_batch` | `(data) -> (loss, detached log_prob)` | `_actor_loss(data.obs)` | `SequenceSAC`, `ACRLPDCore` |
| `SACCore._extra_batch_slice_keys` (class attribute) | `tuple[str, ...]` | `()`; the sample fields `_slice_batch` copies into each high-UTD minibatch, which it rebuilds as `type(batch)(**kwargs)`: a sample field with no default that is not listed here fails the slice with a `TypeError`. Extend with `(*self._extra_batch_slice_keys, "field")` | `SAC` (`"discounts"` when `nstep > 1`), `RLPDHybrid` (`"grasp_penalty"`), `ACRLPDCore`, `CalQLCore` |
| `PriorDataReplayMixin._build_prior_data_buffer` | `(buffer_size) -> buffer` | a single-env buffer, sized independently (`NStepReplayBuffer` when `nstep > 1`, else `ReplayBuffer`); used by `load_offline_replay_buffer` and `DemoInterventionMixin.init_demo_buffer`. It must sample the same dataclass as `replay_buffer`, because the two samples are concatenated | `RLPDHybrid` (adds the `grasp_penalty` column under `use_grasp_penalty`), `ACRLPDCore`, `DAgger` |
| `DemoInterventionMixin.init_demo_buffer` / `add_demo_transition` | `(buffer_size, demo_data_ratio=0.5)` / `(obs, next_obs, action, reward, done, **extra)` | `init_demo_buffer` builds an empty buffer through `_build_prior_data_buffer` into the `offline_replay_buffer` slot and sets `offline_data_ratio` (it raises if the slot is already populated); `add_demo_transition` forwards to that buffer's `add`, `**extra` included. The demo buffer is a sidecar buffer (`_sidecar_buffers`, section 4): saved as `demo_buffer_<suffix>.pt` and named at `state["extra"]["demo_buffer_path"]`; once restored from a file, its running episode is cut (`cut_running_episodes`), so the rows added next start another episode. A loop that resets the env itself while it fills the demo buffer cuts it as it cuts the replay buffer | `RLPDHybrid`, `DAgger` (mixin users) |

### Action chunking

`ActionChunkingMixin` is the leftmost base of `QAM` and `QGF`;
`ChunkedRolloutMixin(ActionChunkingMixin)` is the leftmost base of `ACRLPD`
and `ACFQL`. Call `_init_action_chunking(horizon_length)` before model setup.
The shared `_policy_action_space` tiles the raw flat Box bounds,
`_chunked_replay_buffer(*, num_envs, buffer_size)` keeps storage and sample
devices explicit, and `_build_replay_buffer` uses the host's dimensions.
`_sample_train_batch` calls the next sampler in the MRO, then replaces only
`actions` with its `(B, H*A)` reshape. `_checkpoint_metadata` adds
`horizon_length`; the other hyperparameters stay in the host.
`ChunkedRolloutMixin._sample_action_chunk` predicts a stochastic flat chunk
and reshapes it to `(B, H, *action_shape)`. Rollout and evaluation queues are
separate and allocated by their reset hooks. ACRLPD rejects image observations
in `_setup_model` before calling the SAC setup; its SAC-specific losses,
prior-buffer hook and Q-vs-MC decision-step handling remain in `ACRLPDCore`.

### Metrics

`OnPolicyAlgorithm` logs every key of `train()`'s dict as `losses/<key>` at
`--log-freq`. To log a key elsewhere, override `_log_update_metrics`, pop it, call
`super()` with the rest and then `self._log_rollout_metric(key, value, step)`
(`train/<key>`) or `self.logger.add_scalar(...)`, as the reference package
does for `aux_logit_mean`. SAC-family algorithms return extra info tensors
from `_critic_loss` and `_post_actor_update` (reduced with one host sync per
update, only when `compute_info`).

`.item()` placement: keep host syncs out of per-minibatch code. Collect detached
tensors per minibatch and reduce them once per update in
`_extra_update_metrics` or `train()` (the reference package does the latter
for its `aux_logit_mean` metric).

## 4. Checkpoints

- `save()` records `algorithm_class = type(self).__name__`; `load()` always
  accepts the class's own name, so a subclass needs no declaration to reload
  its own checkpoints. `_compatible_checkpoint_algorithms` lists only OTHER
  class names a checkpoint may carry (e.g. the parent's, to load a parent
  checkpoint into the subclass).
- `_checkpoint_metadata()` returns `{**super()._checkpoint_metadata(), ...}`;
  the extra keys are stored in `metadata["hyperparameters"]`, restored onto
  Args fields of the same name at evaluation, and not validated on load (a
  resume-consistency check goes in `_on_checkpoint_read`, below).
- `save()` writes no absolute path and no command line: whatever the hooks here
  return, a string that is or contains an absolute path is replaced by
  `REMOVED_PATH` (`rl_garden.common.checkpoint`; `holds_removed_path(value)` tests
  for it). It is a string, not `None`, so "was set" stays readable. Code that
  reads a path back from `hyperparameters`, `run_config` or its own
  `_extra_checkpoint_state` must treat it as unknown. See
  [Absolute paths](checkpoint.md#absolute-paths).
- User-supplied policy modules are recorded as import paths under
  `actor_factory`, `actor_extractor`, and `critic_extractor` in
  `metadata["hyperparameters"]`; evaluation restores them before construction.
  A lambda, local or `__main__` definition warns and records an unrecorded
  sentinel, so automatic rebuilding raises `ConfigError`. Extractor kwargs
  are not recorded; prefer an importable class with constructor defaults.
- Extra modules attached to the policy are part of its state dict; modules
  outside it go through `_extra_checkpoint_state` / `_load_extra_checkpoint_state`
  and extra optimizers through `_optimizer_names`.
- `Off2OnReplayMixin` records the online stage in `_extra_checkpoint_state` and
  restores it in `_load_extra_checkpoint_state` without calling
  `_apply_online_regularizer_override`: a subclass that changes its own state at
  the switch restores that state in `_load_extra_checkpoint_state`, where
  `_online_start_step` and the stage are already set after `super()`. A stage
  the checkpoint does not record is stated with `declare_online_replay_mode`
  (below), so a subclass with state that depends on the stage extends both, as
  Cal-QL does for its replay buffer's class. The offline buffer of the `mixed`
  mode is a sidecar buffer, built by `_build_replay_buffer` through
  `restore_offline_replay_buffer`.
- `Off2OnReplayMixin.train` returns `{}` without calling `super().train()` while
  the warmup's update mask allows neither actor nor critic updates, so a
  subclass's `train` is not called during the warmup.
- `save()` and `load()` are not overridden to add files or checks; the seams
  below do that. `load()` reads the checkpoint file once, validates its
  metadata, calls `_on_checkpoint_read`, restores the state
  (`load_state_dict`, which calls `_load_extra_checkpoint_state`), reads the
  replay file and then the sidecar files. A file name the checkpoint records must
  exist (`FileNotFoundError`); `None` recorded means no file was written.

| Seam | Signature | What the base does | In-tree overriders |
|---|---|---|---|
| `BaseAlgorithm._sidecar_buffers` | `() -> tuple[SidecarBuffer, ...]` | `()`. `SidecarBuffer(prefix, reference_key, buffer, restore, unrecorded_beside_replay=False)` (`rl_garden.common.checkpoint`). `save(include_replay_buffer=True)` writes each `buffer` that is not `None` to `<prefix>_<suffix>.pt` beside the checkpoint and names the file at `reference_key` (a key path under `state["extra"]`, whose mappings above the leaf must exist; the subclass's `_extra_checkpoint_state` writes the slot as `None`). `load(load_replay_buffer=True)` calls it again after the state is restored and calls `restore(path, strict)` for each named file; without the slot it reads the `<prefix>_...` file named like the checkpoint's replay file when `unrecorded_beside_replay`, and warns when that file is missing beside an existing replay file. Extend with `(*super()._sidecar_buffers(), SidecarBuffer(...))` | `Off2OnReplayMixin` (`offline_buffer`, in the `mixed` stage), `DemoInterventionMixin` (`demo_buffer`, after `init_demo_buffer`) |
| `BaseAlgorithm._on_checkpoint_read` | `(checkpoint, *, replay_path) -> None` | no-op; called once per `load()` after the file is read and `validate_checkpoint_metadata` passed, before `load_state_dict` and before any buffer file is read. `replay_path` is the replay file `load()` is about to read, `None` when it reads none. An override calls `super()` first, raises to reject the checkpoint (also at `strict=False`), and may add to `checkpoint` what an older checkpoint does not record | `FQLCore` (rejects a `reference_compatible` mismatch), `Off2OnReplayMixin` (`replay_buffer_restored`, the warmup an unrecorded stage's metadata records), `_CalQLRolloutTrainingShell` (an unrecorded stage's replay buffer takes its file's class) |
| `Off2OnReplayMixin.declare_online_replay_mode` | `(online_replay_mode, offline_data_ratio=0.0) -> None` | states the replay mode of a loaded online-stage checkpoint that does not record it (`online_replay_mode is None` after `load()`). The off2on runner calls it with the mode of the run that wrote the checkpoint (its run configuration) or with the explicit flags | `_CalQLRolloutTrainingShell` (gives a replay buffer not restored from a file the mode's class) |
| `Off2OnReplayMixin.replay_buffer_restored` (attribute) | `bool` | whether the last `load()` read a replay file into `replay_buffer` | none |
| `Off2OnReplayMixin.lacks_offline_data` (property) | `bool` | a loaded `mixed` stage with a nonzero ratio that had offline data but has no offline buffer: training raises until `restore_offline_replay_buffer()` is filled | none |
| `Off2OnReplayMixin.restore_offline_replay_buffer` | `() -> buffer` | builds an empty offline buffer with `_build_replay_buffer`, puts it in `offline_replay_buffer` and returns it for the caller to fill | none |

## 5. Evaluation

`rl-garden-eval --checkpoint-path <run>/checkpoints/final.pt` (defaults
`--phase auto --algorithm auto`):

1. picks the registry entry the checkpoint's embedded run configuration
   selected (`selection.training_phase` / `selection.algorithm`). Without a
   selection, or when it names no registered entry, it is the entry whose
   `algorithm_cls` has the checkpoint's class name (the registering module is
   imported through the entry point); a class registered offline and off2on (in
   tree, `WSRL`) resolves by the stage the checkpoint records
   (`state["extra"]["online_start_step"]`: `None` is the offline stage). It
   exits when no entry or several entries match; `--phase` and `--algorithm`
   override each step. The agent's `training_stage` (`Off2OnReplayMixin`) is
   part of the printed result and the JSON;
2. rebuilds the Args: the entry's Args class, the run configuration (including
   the env backend's sub-config), then the checkpoint's hyperparameters, then
   the eval CLI. The run configuration is the file `--config-path` names, else
   `metadata["run_config"]`, else the `config.json` of the run directory. The
   phase runners set it on the agent (`BaseAlgorithm.run_config`, plain JSON
   data) and `save` writes it without its absolute paths and command line,
   together with the hash of the complete one
   (`metadata["run_config_digest"]`), by which evaluation recognizes the run's
   `config.json` beside the checkpoint and reads the paths there. A field whose
   path is in neither keeps its Args default, and is an error when it belongs to
   the environment. A package with its own run function sets the
   attribute itself to make its checkpoints self-contained. Fields the Args
   class does not have are ignored. Extra Args fields round-trip; `--env-backend`, `--env-id`,
   `--control-mode` and `--obs.*` change the run's environment only when given;
   the entry's `training_only_inputs` are set to their absent values;
3. creates one evaluation environment, builds the agent with the entry's
   `build_agent` and loads the weights (`strict` by default). An online or
   off2on builder receives that environment as both `env` and `eval_env`: it
   and the constructor may read its spaces and `num_envs` but not step it;
4. steps `agent.eval_env`, so a builder that wraps the eval env it is given
   (action chunking, for example) is evaluated on the wrapped one, or the
   backend's own eval env when the agent holds none. `--max-eval-steps` counts
   steps of that env, so one covers a whole action chunk.

Evaluation needs the checkpoint and the environment; everything the policy acts with must be in the checkpoint. A
network that came from another file belongs in `_extra_checkpoint_state`, with
what rebuilds it, as `SUPE` (the OPAL VAE) does.
When the builder itself needs it, `rl_garden.training.inspection` offers
`checkpoint_to_restore(args)` (the evaluated or `--load-checkpoint` checkpoint,
`None` for a fresh run) and `current_evaluation()` (`None` unless the agent is
being rebuilt for evaluation).

`--device` reaches the whole build: `construct_agent` passes it to every
constructor that takes `device` (directly or through `**kwargs` to a parent),
also over a `device` the builder passes, and `get_device("auto")` resolves to it.
Replay buffers are built on the `meta` device, without storage;
`--buffer-device cpu` gives real ones, for a custom buffer that cannot be built
that way.

An algorithm whose policy cannot act in the environment evaluation builds (a
pretraining stage, say) sets the class attribute
`checkpoint_eval_unsupported_reason` to a sentence saying why; evaluation then
exits with it before building an environment.

`rl_garden.training.evaluation.evaluate_checkpoint(EvalCheckpointArgs(...))`
is the same thing as a function. See [Checkpoint Save & Load](checkpoint.md).

## 6. RLinf

The RLinf adapters select classes by config name. Register yours at import
time, in a module every RLinf worker process imports:

- `rl_garden.integrations.rlinf.ppo_model.register_policy(name, policy_cls)`
  for `cfg.actor.model.rlgarden_policy` (PPO contract policies);
- `rl_garden.integrations.rlinf.sac_actor.register_algorithm(name, cls)` and
  `rl_garden.integrations.rlinf.offline_actor.register_algorithm(name, cls)`
  for `cfg.actor.model.rlgarden_algorithm`.

A taken name raises `ValueError`. See
[RLinf Integration](../design/rlinf-integration.md) for the contracts.

(extending-env-backends)=

## 7. Environment backends

A registered single-instance `gymnasium.Env` needs no backend
([Training on your own gymnasium envs](gymnasium-custom-envs.md)). A backend is a
class that turns a backend-neutral `EnvRequest` into a vector env; the API is
`rl_garden.envs.backend_registry`.

### Registration

```python
from rl_garden.envs.backend_registry import EnvBackend, EnvRequest, register_env_backend


class MyBackend(EnvBackend):
    api_version = 2
    config_field = "my_backend"

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        from my_package.env import MyEnvConfig

        return MyEnvConfig(
            env_id=req.env_id,
            num_envs=req.num_eval_envs if is_eval else req.num_envs,
            seed=req.seed,
            reward_scale=req.reward_scale,
            reward_bias=req.reward_bias,
        )

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        from my_package.env import make_env

        return make_env(cls.resolve_config(req, is_eval=False))

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        from my_package.env import make_env

        return make_env(cls.resolve_config(req, is_eval=True))


register_env_backend("my_backend", MyBackend)
```

`register_env_backend(name, cls)` raises `ValueError` for a taken name and
`TypeError` unless all of these hold:

- `api_version = 2` is set in the class's own body;
- `config_field` is a `str` set in the class's own body and equals `name`;
- `resolve_config`, `make_train_env` and `make_eval_env` are in the class's own
  `__dict__`. Inherited methods do not count: a subclass of an existing backend
  defines all three, delegating to `super()` where it changes nothing.

An installed package declares the module that registers the backend, which then
is selectable with `--env_backend my_backend`:

```toml
[project.entry-points."rlgarden.env_backends"]
my_backend = "my_package.backend"
```

The registry imports the in-tree backends and then every module of this group,
once per process, the first time a backend is looked up. `--print-config` already
looks one up, so import the simulator inside the methods, not at module level. A
package that is not installed imports its backend module from its own launcher
before `registry.run_cli()`.

### Methods

| Method | Called |
|---|---|
| `config_from_args(args)` | by preflight and whenever an `EnvRequest` is built; the result becomes `req.backend_config`. The default returns `getattr(args, config_field)` and raises `ValueError` when Args has no such field. Override it to read an existing section (see [Backend-specific named knobs](gymnasium-custom-envs.md#backend-specific-named-knobs), also for adding a section) |
| `resolve_config(req, *, is_eval)` | when a run or `--dry-run` is materialized, in addition to your own calls, once for train and, if `req.create_eval_env`, once for eval. Returns the concrete config that `config.json` records; it must not create an environment |
| `make_train_env(req)` | once per online or off2on run. Offline runs and checkpoint evaluation never call it |
| `make_eval_env(req)` | when `req.create_eval_env`, for an offline run's live evaluation, and for checkpoint evaluation of every phase, which creates only this env. A backend that cannot host a second env raises `NotImplementedError`; its runs then need `--eval_freq 0` |

Under DDP an online run needs a dataclass `req.backend_config`: each rank
rewrites its string fields named `device` or `sim_device` that hold `cuda*`, and `sim_backend` or
`render_backend` that hold `gpu*`, to its own GPU before `make_train_env`.

### `EnvRequest`

| Field | Meaning |
|---|---|
| `env_id`, `control_mode`, `render_mode`, `seed` | the CLI values |
| `num_envs` / `num_eval_envs` | number of train / eval envs; the same request serves both, `is_eval` says which |
| `observation` | the `ObservationConfig` to honor, below |
| `reward_scale`, `reward_bias` | the reward transform, below |
| `create_eval_env` | whether an online or off2on run builds an eval env |
| `num_eval_steps` | env steps per evaluation |
| `eval_record_dir`, `capture_video`, `video_fps` | eval video: recorded only when `capture_video` is set, into `eval_record_dir` (`None` when there is none) |
| `backend_config` | the `config_from_args` result; `None` in a request built without one |

**Reward.** Nothing outside the backend applies `reward_scale` and
`reward_bias` to live rewards: `step()` of the training env returns
`reward_scale * reward + reward_bias`. `RewardScaleBiasWrapper` does this;
`apply_reward_transform(env, scale=..., bias=...)`
(`rl_garden.envs.wrappers.reward_transform`) adds it unless the transform is the
identity. Its place in the stack is fixed by two rules:

- outside the episode statistics, so that the reported episode `return` stays the
  sum of raw rewards;
- once per `step()` of the env the backend returns: outside the backend's own
  action repeat, inside action chunks and skills, which the algorithm adds. The
  offline loaders apply the same transform once per stored transition, and one
  `step()` of the returned env is one transition, so a bias such as `-1` means the
  same online and offline.

The order is fixed because it changes the reward: `ActionRepeatWrapper`,
`ActionChunkWrapper` and `SkillActionWrapper` sum rewards, so a transform inside
them adds the bias once per counted step and one outside adds it once per sum. Frame
stacking does not touch the reward. In-tree stacks, innermost first:

| Backend | Stack | Bias added |
|---|---|---|
| gymnasium, ogbench, rlbench, metaworld | adapter (gymnasium: then `obs_key_map`), transform, frame stack | per env step |
| dm_control | single env with its action repeat, adapter, transform, frame stack | per decision |
| mujoco_warp | task, transform, frame stack | per env step |
| custom, mujoco, d4rl_legacy, robomimic | adapter, transform | per env step |
| maniskill | frame stack, `ManiSkillVectorEnv`, transform | per env step |
| isaaclab | adapter, `ActionRepeatWrapper`, frame stack, transform | per decision |
| minari | adapter | no transform |

A builder that adds `ActionChunkWrapper` or `SkillActionWrapper` puts it outside all
of these, so a chunk contains the bias once per counted `step()` of the backend's env
(on isaaclab and dm_control, per decision).

### Observation contract

Right after construction the registry runs `validate_observation_space` on
`env.single_observation_space` and requires its keys to equal
`req.observation.expected_keys` exactly; otherwise it raises
`ObservationContractError`. The keys are `rgb_<cam>` and `depth_<cam>` per
requested camera, `state` unless `--obs.no-state`, and `state_<name>` per
`extra_state` name; shapes and dtypes are those in
[Observation keys](gymnasium-custom-envs.md#observation-keys), and an image may
have a leading frame-stack dimension. The check compares key names, not
`image_size` or `frame_stack`: a backend honors those too, or raises
`ObservationContractError` itself. A state-only backend calls
`require_state_only_observation(req.observation, backend="my_backend")`
(`rl_garden.envs.wrappers`) in `resolve_config`.

### Env surface

The returned env has `num_envs`, `single_observation_space`,
`single_action_space` and the batched `action_space`. `reset(seed=...)` returns
`(obs, infos)` and `step(actions)` returns `(obs, rewards, terminations,
truncations, infos)`, with torch tensors (observations a dict of them) on the
env's device. An env that resets itself inside `step()` sets
`metadata["autoreset_mode"]` to `AutoresetMode.SAME_STEP` and reports an ended
episode in `infos`:

- `final_info["episode"]`: per-env tensors, `return` and `episode_len` (or
  gymnasium's `r` and `l`), optionally `success_once` and `success_at_end`;
- `_final_info`: the `(num_envs,)` bool mask of the rows that ended, required
  with `final_info`;
- `final_observation`: like `obs`, with the terminal observation in the ended
  rows. Without it on-policy algorithms bootstrap an ended episode with 0 and
  off-policy algorithms store the returned post-reset observation as the
  successor.

`infos["decision"]` and `infos["decision_components"]` are optional, see
[Decision-Level PPO](decision-level-ppo.md#2-env-contract).
`rl_garden.envs.vector_env.TorchVectorEnvAdapter` provides this surface for a NumPy
`gymnasium.vector.VectorEnv` created with `SAME_STEP` whose single envs are wrapped
in `gymnasium.wrappers.RecordEpisodeStatistics`.
`tests/test_vector_env_contract.py` and
`tests/test_episode_boundary_contract.py` check it on in-tree adapters and envs.

### Wrapping a batched env

At and below the NumPy-to-torch adapter, and directly on a torch-native vector env,
wrappers are gymnasium's: `gymnasium.Wrapper` on a single env,
`gymnasium.vector.VectorWrapper` on the vector env (`TorchVectorEnvAdapter` itself;
`SeededResetVectorEnv` in the same module, which makes `reset()` without a seed
replay the configured one). Every wrapper above that derives from
`BatchedEnvWrapper` (`rl_garden.envs.wrappers.batched_env`), accepts any batched env
and so stacks in any order with the others. The orders are not equivalent: see the
reward rules above.

`BatchedEnvWrapper` is a `gymnasium.Wrapper` that accepts both a batched
`gymnasium.Env` and a `gymnasium.vector.VectorEnv`, and it forwards
every attribute it does not define (`num_envs`, `device`, the `single_*_space`s,
`post_update_sync`, ...) to the wrapped env on each access, so do not copy them. A
subclass that changes the observation or action contract sets both the
`single_*_space` and the batched space; otherwise callers read the wrapped env's.
Writes do not forward: an assignment, also through `set_wrapper_attr`, stays on the
wrapper. The wrapped result is not a `VectorEnv`, so do not test
`isinstance(env, VectorEnv)` on an env a backend returns, and wrap it in
`VectorWrapper`s only below the first `BatchedEnvWrapper`.

In-tree subclasses: `RewardScaleBiasWrapper`, `ImageFrameStackWrapper`,
`ActionRepeatWrapper`, `ActionChunkWrapper`, `SkillActionWrapper`,
`MultitaskPadWrapper`, `GAILRewardWrapper`, `VectorizedDictStateWrapper`,
`RewardClassifierWrapper`, `MultiStageBinaryRewardClassifierWrapper` and
`FWBWResetFreeWrapper`.
