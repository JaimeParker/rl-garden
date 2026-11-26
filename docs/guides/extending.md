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

### Constructor ordering

`PPO.__init__` and `SAC.__init__` end by calling `self._setup_model()`, which
builds the policy. Attributes your overrides read must be set **before**
`super().__init__()`, as `RecurrentPPO` does with its `rnn_*` fields:

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

The parent constructors take a fixed keyword list (no `**kwargs` catch-all):
forward the rest through `**kwargs` as above. A subclass whose policy replaces
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
| `OfflineSAC.policy_class` (class attribute) / `OfflineSAC._policy_kwargs(observation_space) -> dict` | `SACPolicy`; `observation_space`, `action_space`, `net_arch`, `n_critics`, `critic_subsample_size`, `actor_factory` (only when given), then the extractors | as `CQLCore`'s (`EDAC`, `SO2`) |
| `actor_factory` (constructor keyword of `SAC`, `OfflineSAC`, `CQL`, `CalQL`, `WSRL` and their off2on shells; class attribute on `SACCore`, `None`) | `fn(features_dim, action_space) -> SACActor` that the default `SACPolicy._build_actor` calls; recorded in the checkpoint as an import path | `SACFlow` and `SequenceSAC` reject it (their policies build their own actor) |
| `SACPolicy._build_actor(features_dim, action_space, **actor_options) -> nn.Module` | `SquashedGaussianActor` with the policy's Gaussian options, or `actor_factory(features_dim, action_space)` (options not forwarded) | called once, right after the critic and `critic_target` (the RNG position of the old inline build), with the actor's feature width (`actor_feature_dim`'s compressor output when set); a subclass sets its own options as plain attributes before `super().__init__()`. The result is validated (`validate_sac_actor`) and its flags stored as `policy.actor_capabilities` |
| `SACCore._check_actor_capabilities(capabilities)` | `ValueError` for `ent_coef="auto"` with `stochastic=False`; `CQLCore` adds `cql_importance_sample=True` with `log_prob_is_density=False`, `ACRLPDCore` `bc_alpha > 0` with `supports_evaluate_log_prob=False` | runs once in `_setup_model` right after the policy is built; extend with `super()._check_actor_capabilities(capabilities)` |
| `SACCore._target_policy_action(next_action) -> Tensor` | `next_action` unchanged when `target_policy_noise` is 0 (no random numbers), else plus clipped Gaussian noise (`target_noise_clip`), clamped to the action bounds | the next action the critic target reads (TD3's target smoothing); `SO2Core` keeps its own smoothing |
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
table and `tests/test_off_policy_seams.py` in the same commit. `learn()` runs,
per environment step: `_rollout_action`, `env.step`, `_on_env_step`,
`_replay_buffer_add_kwargs`
and `_replay_buffer_step_kwargs` (merged into one `replay_buffer.add(obs,
real_next_obs, replay_action, rewards, stop_bootstrap, **kwargs)` call), then
`_post_rollout_step`.

| Hook | Signature | What the base does | In-tree overriders |
|---|---|---|---|
| `OffPolicyAlgorithm._rollout_action` | `(obs, learning_has_started: bool) -> (replay_action, env_action, action_context)` | `_policy_action(obs)` mixed with `_explore_action(obs)` at `random_action_prob` while an initial training phase is active; otherwise `_explore_action` before `learning_starts` and `_policy_action` after; returns `(actions, actions, None)`. `replay_action` is stored in the buffer, `env_action` goes to `env.step`, `action_context` (a dict or `None`) reaches `_replay_buffer_add_kwargs` and `_post_rollout_step` of the same step | `DDPG`, `FlashSAC`, `SequenceSAC`, `JSRL`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `NCRL`, `SuperMimic`, `_SPOTRolloutTrainingShell` |
| `OffPolicyAlgorithm._explore_action` | `(obs) -> Tensor` | uniform in `[-1, 1]`, shaped like `env.action_space` (all envs), on the algorithm's device | `NCRL`, `SuperMimic` |
| `OffPolicyAlgorithm._on_env_step` | `(rewards, terminations, truncations) -> None` | no-op; called right after every training `env.step` with the tensors it returned, before the replay write | `FlashSAC` (updates its reward normalizer) |
| `OffPolicyAlgorithm._replay_buffer_add_kwargs` | `(action_context, obs, next_obs, real_next_obs, infos, need_final_obs) -> dict` | `{}`; the dict is passed as extra keywords to `replay_buffer.add` | `RLPDHybrid` (`grasp_penalty` from `infos`), `SequenceSAC`, `TransformerSAC`, `DreamerV3`, `SuperMimic` |
| `OffPolicyAlgorithm._replay_buffer_step_kwargs` | `(terminations, truncations) -> dict` | `{}` (`SAC`: `{"episode_end": terminations \| truncations}` when `nstep > 1`); merged after the add keywords, so a key present in both takes this value | `SAC`, `DDPG`, `FlashSAC`, `SequenceSAC`, `RLPDHybrid`, `ACRLPDCore`, `_CalQLRolloutTrainingShell`, `ModelBasedAlgorithm` |
| `OffPolicyAlgorithm._on_env_reset` | `(obs) -> None` | no-op; called once by `learn()` right after its `env.reset(seed=...)` (later resets happen inside `env.step`, see `terminations` / `truncations` in `_post_rollout_step`) | `SequenceSAC`, `JSRL`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `NCRL`, `SuperMimic` |
| `OffPolicyAlgorithm._post_rollout_step` | `(action_context, terminations, truncations, infos) -> None` | no-op; runs after the replay write and before `obs = next_obs` | `SequenceSAC`, `JSRL`, `ChunkedRolloutMixin`, `TDMPC2`, `DreamerV3`, `SuperMimic` |
| `BaseAlgorithm._eval_action` | `(obs) -> Tensor` | `policy.predict(obs, deterministic=True)` under `no_grad`, observations moved to the policy device | `JSRL`, `PWM`, `PolicyDistillation`, `SuperMimic`, `OnPolicyAlgorithm` |
| `BaseAlgorithm._eval_action_and_critic_action` | `(obs) -> (env_action, critic_action)` | `(a, a)` with `a = _eval_action(obs)`; the first is executed, the second is what the critic scores | `SequenceSAC`, `JSRL`, `DreamerV3`, `NCRL`, `SequencePPO` |
| `BaseAlgorithm._eval_start_hook` | `() -> None` | no-op; called once after `eval_env.reset()`, before the eval loop (`SACCore` resets its Q-vs-MC bookkeeping here) | `SACCore`, `SequenceSAC` (does not call `super()`), `JSRL`, `TDMPC2`, `DreamerV3`, `SuperMimic`, `SequencePPO` |
| `BaseAlgorithm._eval_step_hook` | `(obs_before, critic_action, rewards, terminations, truncations, infos) -> None` | no-op; called after every eval `env.step` (`SACCore` accumulates Q-vs-MC when `q_mc_diagnostics`) | same classes as `_eval_start_hook` |
| `SAC._build_replay_buffer` | `() -> BaseReplayBuffer` | `NStepReplayBuffer` when `nstep > 1`, else `ReplayBuffer` over `num_envs` envs; called once in `_setup_model` | `RecurrentSAC`, `TransformerSAC`, `RLPDHybrid`, `ACRLPDCore`, `QAMCore`, `QGFCore`, `SO2Core`, `CalQLCore` |
| `SAC._policy_action_space` | `() -> spaces.Box` | `env.single_action_space`; read by `_policy_kwargs` and by the buffer builders of the overriders | `ACRLPDCore`, `QAMCore`, `QGFCore`, `ACFQLCore` |
| `SACCore._sample_train_batch` | `(batch_size) -> sample` | `replay_buffer.sample(batch_size)`; called once per gradient step (once per group under high UTD). The sample is a dataclass with `obs`, `next_obs`, `actions`, `rewards`, `dones` and the fields named by `_extra_batch_slice_keys` | `PriorDataReplayMixin` (mixes in the prior-data buffer), `ExPLORe`, `Off2OnReplayMixin`, `ACRLPDCore`, `QAMCore`, `QGFCore`, `OfflineSAC`, `CQL` |
| `SACCore._target_action_log_prob` | `(data) -> (next_action, next_log_prob, next_actor_features)` | `_actor_action_log_prob(data.next_obs)`; the TD target's next action under the current actor | `SO2Core` |
| `SACCore._actor_loss_from_batch` | `(data) -> (loss, detached log_prob)` | `_actor_loss(data.obs)` | `SequenceSAC`, `ACRLPDCore` |
| `SACCore._extra_batch_slice_keys` (class attribute) | `tuple[str, ...]` | `()`; the sample fields `_slice_batch` copies into each high-UTD minibatch, which it rebuilds as `type(batch)(**kwargs)`: a sample field with no default that is not listed here fails the slice with a `TypeError`. Extend with `(*self._extra_batch_slice_keys, "field")` | `SAC` (`"discounts"` when `nstep > 1`), `RLPDHybrid` (`"grasp_penalty"`), `ACRLPDCore`, `CalQLCore` |
| `PriorDataReplayMixin._build_prior_data_buffer` | `(buffer_size) -> buffer` | a single-env buffer, sized independently (`NStepReplayBuffer` when `nstep > 1`, else `ReplayBuffer`); used by `load_offline_replay_buffer` and `DemoInterventionMixin.init_demo_buffer`. It must sample the same dataclass as `replay_buffer`, because the two samples are concatenated | `RLPDHybrid` (adds the `grasp_penalty` column under `use_grasp_penalty`), `ACRLPDCore`, `DAgger` |
| `DemoInterventionMixin.init_demo_buffer` / `add_demo_transition` | `(buffer_size, demo_data_ratio=0.5)` / `(obs, next_obs, action, reward, done, **extra)` | `init_demo_buffer` builds an empty buffer through `_build_prior_data_buffer` into the `offline_replay_buffer` slot and sets `offline_data_ratio` (it raises if the slot is already populated); `add_demo_transition` forwards to that buffer's `add`, `**extra` included | `RLPDHybrid`, `DAgger` (mixin users) |

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
  resume-consistency check is the subclass's own `load` override).
- User-supplied policy modules are recorded as import paths under
  `actor_factory`, `actor_extractor`, and `critic_extractor` in
  `metadata["hyperparameters"]`; evaluation restores them before construction.
  A lambda, local or `__main__` definition warns and records an unrecorded
  sentinel, so automatic rebuilding raises `ConfigError`. Extractor kwargs
  are not recorded; prefer an importable class with constructor defaults.
- Extra modules attached to the policy are part of its state dict; modules
  outside it go through `_extra_checkpoint_state` / `_load_extra_checkpoint_state`
  and extra optimizers through `_optimizer_names`.

## 5. Evaluation

`rl-garden-eval --checkpoint-path <run>/checkpoints/final.pt` (defaults
`--phase auto --algorithm auto`):

1. picks the registry entry whose `algorithm_cls` has the checkpoint's class
   name (the registering module is imported through the entry point); it exits
   when no entry or several entries match, so an ambiguous class needs `--phase`
   (in tree, `WSRL`) and one no entry declares needs `--algorithm` (`SACDDP`);
2. rebuilds the Args: the entry's Args class, the run configuration (including
   the env backend's sub-config), then the checkpoint's hyperparameters, then
   the eval CLI. The run configuration is the file `--config-path` names, else
   `metadata["run_config"]`, else the `config.json` of the run directory. The
   phase runners set it on the agent (`BaseAlgorithm.run_config`, plain JSON
   data) and `save` writes it; a package with its own run function sets the
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
what rebuilds it, as `SUPE` (the OPAL VAE) and `JSRL` (the guide policy) do.
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
