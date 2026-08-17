# RLinf Integration

## Status

Three adapters are built, tested, and verified end-to-end against a real
RLinf/Ray/GPU cluster. This document is a record of what actually works and
why, plus a design guide for keeping future rl-garden algorithms and
network code easy to plug into RLinf — not a proposal. Earlier drafts of
this document speculated about algorithms that were never built (DDPG
contract, RLPDHybrid, Cal-QL under RLinf, RecurrentSAC/TransformerSAC,
SequencePPO, TDMPC2); that speculation has been removed. What remains
below is either shipped code or an explicit, short "not built" list.

| Adapter | Algorithms | RLinf runner | FSDP? | Code |
|---|---|---|---|---|
| Offline | IQL, BC, CQL, AWAC, TD3+BC | `OfflineRunner` (static) | No — plain `Worker` | `rl_garden/integrations/rlinf/{offline_actor,train_offline}.py` |
| Online off-policy | SAC, RLPD | `AsyncEmbodiedRunner` | No — plain `Worker` | `rl_garden/integrations/rlinf/{sac_actor,sac_rollout,train_sac}.py` |
| Online on-policy | PPO | `EmbodiedRunner` (sync) | **Yes** — `EmbodiedFSDPActor` | `rl_garden/integrations/rlinf/{ppo_model,ppo_actor,ppo_advantages,ppo_rollout,train_ppo}.py` |

All three are lazily-imported optional dependencies (`try: import rlinf
... except ImportError:`), so core `rl_garden` imports, registry discovery,
and the non-RLinf test suite work with RLinf uninstalled — the same
convention `AGENTS.md` requires for other optional backends. `3rd_party/
RLinf` is a read-only vendored reference copy; nothing under it is edited
by any of this.

Verification records (exact launch commands, GPU/cluster state, every bug
hit and its fix) live in the machine-local, gitignored `.agents/local/
6017.md` and `9990.md` — this document explains the *shape* of what was
built and *why*, those files have the blow-by-blow.

## Why RLinf, and the two adapter shapes

RLinf brings orchestration rl-garden does not build itself: Ray-based
multi-process/multi-GPU cluster management, FSDP-sharded training,
distributed rollout collection, weight synchronization between training and
inference processes. rl-garden stays scoped to algorithm math, policies,
encoders, and environment backends. Two genuinely different ways to plug
into RLinf turned out to both be correct, for different reasons — pick
based on what a given algorithm needs, not by default:

**Pattern A — lightweight `Worker`, rl-garden drives its own math.**
Subclass RLinf's plain `Worker` (not any FSDP-specific base class). Build
and train an ordinary rl-garden `nn.Module`/optimizer pair directly; RLinf
supplies only Ray orchestration, data channels, and weight sync.
`rl_garden.algorithms.<Algo>.train()` actually executes. Used for the
Offline and SAC/RLPD adapters. Correct default when an algorithm's own
`train()`/loss math is already correct and you just want RLinf's rollout/
dataloading/cluster machinery around it — inheriting FSDP machinery only to
override it away is the wrong move (see both adapters' module docstrings).

**Pattern B — subclass RLinf's own FSDP worker, RLinf drives training.**
Subclass `EmbodiedFSDPActor` directly to reuse RLinf's FSDP-aware training
loop (shuffle, multi-epoch minibatch passes, gradient accumulation,
distributed metric all-reduce, real multi-GPU parameter sharding) instead
of reimplementing it. The hard consequence: rl-garden's own algorithm
`train()`/loss methods are **not** used — RLinf's own generic loss functions
(`compute_ppo_actor_loss`/`compute_ppo_critic_loss`, algorithm-agnostic, no
LLM/VLA-specific terms) drive training directly, and rl-garden's policy
contributes only network architecture, wrapped to satisfy RLinf's model
protocol and registered through RLinf's own model registry. Checkpointing
also moves to RLinf's own FSDP state-dict format — a checkpoint from this
path is not loadable by rl-garden's own `<Algo>.load()`, and vice versa.
Used for the PPO adapter, verified at both single-GPU (`no_shard`) and real
2-GPU (`full_shard`) sharding. Worth it only when you actually want RLinf's
FSDP multi-GPU training, not merely its orchestration.

Both patterns launch via the same zero-edit mechanism: RLinf's worker
classes are launched via `ray.remote(cls)` on a raw Python class object,
with no `runtime_env`/`working_dir` gate and no assumption that the class
lives under the `rlinf.*` module path. A class defined entirely inside
`rl_garden/integrations/rlinf/` and subclassing an RLinf worker base class
satisfies RLinf's runner contracts with zero RLinf source edits.

## Pattern A in detail: Offline and SAC/RLPD

### Hook contract

| Hook | Purpose |
|---|---|
| `init_worker()` | build the algorithm (via `OfflineEnvSpec`, no live env — see below) |
| `_sample_train_batch(batch_size)` (or buffer swap, see below) | supply training batches |
| `run_training()` → metrics dict | delegate to `<Algo>.train(gradient_steps)` |
| `save_checkpoint`/`load_checkpoint` | delegate to `<Algo>.save()`/`.load()` |
| `sync_model_to_rollout()` | copy weights to the rollout worker (online only) |
| `.worker_group_name` | required attribute |

**No live env needed at construction.** `OfflineEnvSpec`
(`rl_garden/algorithms/offline.py`) exposes `single_observation_space`/
`single_action_space`/`num_envs` with deliberately no `reset()`/`step()`.
Every offline and SACCore-contract algorithm's `_setup_model()` only reads
`.single_observation_space`/`.single_action_space` at construction time —
`.reset()`/`.step()` calls live exclusively inside `learn()`'s rollout
loop, which an RLinf adapter never invokes (RLinf's own worker drives
training). Build an `OfflineEnvSpec` from RLinf-side config-derived spaces
and pass it as `env=`; no `BaseAlgorithm`/`OffPolicyAlgorithm` code changes
were needed to make this work.

**Replay-buffer injection: swap the object, never patch the method.** For
SAC/RLPD, the correct injection point is replacing
`self._algo.replay_buffer` (the object) with a shim wrapping RLinf's own
`TrajectoryReplayBuffer`, *after* `_setup_model()` has run — not
monkey-patching `_sample_train_batch`. `RLPD` overrides
`_sample_train_batch` itself to mix online (RLinf's buffer) and offline
(rl-garden's own) data; patching the method away would silently degrade
RLPD to plain SAC with zero offline data, no exception. Swapping the
buffer object means both `SACCore._sample_train_batch` and
`PriorDataReplayMixin._sample_train_batch` correctly route through it with
zero adapter-side branching between SAC and RLPD. Verified by a regression
test (`tests/test_rlinf_sac_actor.py`) that constructs `RLPD`, swaps the
buffer, and asserts data from *both* the swapped buffer and a real offline
buffer appear in a sampled batch.

**`dones` must come from `terminations`, not the combined field.**
RLinf's `Trajectory.dones = terminations | truncations`, unaffected by
`ignore_terminations` (which only gates episode-metric logging). rl-garden's
own `dones` is specifically the bootstrap-suppression terminal flag — using
the combined field would silently suppress bootstrapping at every
truncation too, not just true episode ends. This distinction was needed
identically in both the SAC/RLPD adapter (`sac_actor.trajectory_batch_to_sample`)
and the PPO adapter's advantage computation — treat it as a standing rule
for any future adapter that reads RLinf trajectory data.

**Generic checkpoint/optimizer hooks already absorb algorithm-specific
extras.** `_optimizer_names()`/`_extra_checkpoint_state()`
(`BaseAlgorithm`, consumed via `getattr` so absent attributes are silently
skipped) already correctly round-trip CQL's Lagrange-multiplier optimizers
and RLPD's extra state with zero adapter-side algorithm-specific code —
verified by driving IQL, BC, and CQL through the same offline adapter code
path with no branching.

### Async off-policy specifics (SAC/RLPD only)

RLinf's async path (`AsyncEmbodiedRunner`) is a genuine off-policy replay
loop: a persistent `TrajectoryReplayBuffer` with a background thread
continuously writing rollout trajectories while training concurrently
samples from it. The rollout worker
(`RLGardenSACRollout(AsyncMultiStepRolloutWorker)`) must return
`forward_inputs["action"]` — RLinf's `EnvWorker` builds trajectory actions
from `policy_output.forward_inputs.get("action", None)`, **not** from
`PolicyOutput.actions` (which is populated from the same top-level return
value but never read back out for trajectory building) — leaving
`forward_inputs["action"]` empty makes every trajectory's actions field
silently drop out during buffer flattening, surfacing several layers away
as a bare `KeyError: 'actions'`. This is exactly the "silent-degradation"
pattern to watch for in any future rollout worker: RLinf reads specific
dict keys generically, not by contract-checked schema, so a missing key
fails loud only if something downstream happens to assert on it.

## Pattern B in detail: PPO on RLinf's own FSDP loop

### The model-protocol wrapper

`RLGardenPPOModel` (`ppo_model.py`) wraps an rl-garden policy (`PPOPolicy`)
to satisfy RLinf's embodied model protocol —
`default_forward(forward_inputs, compute_logprobs, compute_entropy,
compute_values, **kwargs) -> dict` and `predict_action_batch(env_obs, ...)
-> (chunk_actions, result)`, the same shape RLinf's own reference model
(`MLPPolicy`) implements. Registered under a **new** model-type key
(`"rl_garden_ppo"`, via `register_model()`) rather than overwriting RLinf's
existing `"mlp_policy"` — both are zero-RLinf-source-edit
(`register_model`/`SupportedModel.register()` are RLinf's own public
extension points), but a new key has a smaller behavioral footprint on the
shared process. Never references `PPOPolicy` by name outside a small
`_POLICIES` registry, so a future policy on the same duck-typed contract
(`act_with_value_and_logprob`, `evaluate_actions(..., sum_dims=False)`,
`predict_values`) needs only a registry entry, not a wrapper change.

The actor (`RLGardenPPOFSDPActor(EmbodiedFSDPActor)`) overrides exactly one
method, `compute_advantages_and_returns` — bypassing RLinf's own `"gae"`
registry entry (whose own docstring admits it does not support
auto-reset — it bootstraps every step from `values[step+1]`, which under
`auto_reset: True` is the value of the *post-reset* observation) in favor
of rl-garden's own auto-reset-correct GAE, extracted as a standalone
function (`rl_garden/buffers/gae.py::compute_gae`) so it's reusable outside
a `RolloutBuffer` object — see "Design guidance" below for why this
extraction is a pattern worth repeating. Everything else —
`init_worker`/`setup_model_and_optimizer`, `sync_model_to_rollout`,
`recv_rollout_trajectories`, `run_training`/`train_micro_batch` (the full
FSDP training loop), checkpointing — is inherited from `EmbodiedFSDPActor`
unchanged.

### What never gets constructed

This design never instantiates `rl_garden.algorithms.ppo.PPO`, only the
bare `PPOPolicy` network. `PPO`'s own `train()`/GAE loop does not run.
`PPO._optimizer_names()`/`_extra_checkpoint_state()` are not used —
optimizer construction and checkpointing both flow through RLinf's
`FSDPModelManager` instead.

### Five real bugs, and the general lessons behind each

None were guessable from static reading; all five surfaced only by
actually launching against a real Ray/FSDP cluster (see `.agents/local/
6017.md` for the literal error text and fix commits). Each has a lesson
that generalizes past this one adapter:

1. **RLinf's model registry (`register_model`) is a plain in-process
   dict.** Actor and rollout workers each run in their own Ray-spawned
   process; a registration call made only in the driver's `main()` never
   reaches a worker's own copy. **Lesson**: any process-global RLinf-side
   registration must happen at *import time* of a module every relevant
   worker actually imports (Ray re-imports a worker's defining module to
   reconstruct the remote actor class), not via an explicit driver-only
   call — the opposite of Pattern A's convention (`require_rlinf()` as a
   visible call site), which was safe there only because Pattern A never
   registers anything with RLinf's registry at all.
2. **Neither `env_obs` nor a received rollout batch is guaranteed to
   already be on the model's device.** RLinf's own reference model
   (`MLPPolicy.preprocess_env_obs`) explicitly moves inputs to the model's
   device first; a custom model must do the same. **Lesson**: any tensor
   arriving from outside the current process (Ray channel, another
   worker's rollout) needs an explicit `.to(device)`, never an assumption.
3. **Missing config fields** the reference example config carries but a
   trimmed-down config silently dropped (`algorithm.entropy_bonus`/
   `reward_type`/`logprob_type`/`entropy_type`, all bare-attribute reads
   with no `.get()` fallback). **Lesson**: this is the same class of
   gotcha every adapter hits from skipping `validate_cfg` (see the
   gotcha catalog below) — when trimming a reference RLinf config, diff
   against the full reference before removing a field, don't assume a
   field is example-specific just because it looks unrelated to your
   algorithm.
4. **`RLGardenPPOModel(nn.Module, RLinfBasePolicy)`'s `forward()` was
   silently shadowed** by `nn.Module`'s own stub in MRO resolution,
   regardless of what the mixin implements, since `nn.Module` was listed
   first. **Lesson**: don't rely on multiple-inheritance ordering to
   resolve a dispatch method correctly — define `forward()` explicitly on
   the wrapper class itself.
5. **Only visible under real multi-GPU sharding, not `no_shard`**: the GAE
   value-head forward called `self.model.default_forward(...)` directly (a
   bare method call) instead of `self.model(...)` (which triggers
   `nn.Module.__call__`, the hook point FSDP uses to all-gather a sharded
   parameter's full weight before a real forward pass runs). Under
   `no_shard` every rank already holds the complete parameter, so this was
   invisible; under `full_shard` it crashed with a zero-sized storage.
   **Lesson**: once FSDP sharding is genuinely in play (anything beyond
   `no_shard`), *always* invoke an FSDP-wrapped module via `model(...)`,
   never via a bare method call on it — a rule with no exceptions, not a
   case-by-case judgment call.

Also verified as a real, not merely theoretical, constraint:
**`FSDPModelManager.build_optimizer` buckets parameters into a separate
LR group by substring-matching `"value_head"`/`"model.value_head"`
against `model.named_parameters()`.** Since `RLGardenPPOModel` aliases an
already-registered submodule (`self.value_head = policy.value_net`) to
expose that name at the top level, `nn.Module`'s parameter-deduplication
(`remove_duplicate=True` by default) keeps whichever *registration order*
came first when the same tensor is reachable via two attribute paths —
confirmed empirically. `value_head` must be assigned *before* `policy` in
`__init__`, or the substring never appears in the deduped output and the
value head silently trains at the actor's learning rate instead of its own.

## Verified integration mechanisms (both patterns)

- **Launch-plane class launching**: `ray.remote(cls)` on any importable
  class, no `rlinf.*` namespace requirement — the mechanism both patterns
  rely on to run rl-garden code inside RLinf worker processes.
- **`register_model`/`SupportedModel.register()`**: RLinf's model registry
  is a real, public, mutation-based extension point (Pattern B only).
- **`Trajectory.forward_inputs`**: an open `dict[str, Any]` carrier that
  survives the full rollout→training pipeline generically (RLinf's own
  trajectory builder appends it, stacks/splits it, and unions it across
  trajectories with no key-name awareness). Used in both shipped adapters
  for the `"action"`/`"states"` round-trip. **Not yet exercised** for its
  originally-scoped purpose — threading opaque recurrent hidden state for
  a not-yet-built sequence-model on-policy adapter — but the mechanism
  itself (verified: RLinf's own NFT diffusion-policy worker already
  stashes unrelated per-timestep state there) is real and available should
  that be built later. Don't treat "not yet used for X" as "doesn't work
  for X" — it's simply unbuilt, not unverified as a mechanism.

## Gotcha catalog: `validate_cfg` is skipped, every hand-written config pays for it

Every adapter's entry script deliberately skips RLinf's `validate_cfg`
(it rejects any `model_type` outside RLinf's own registered enum, which a
custom-registered model or a non-RLinf-model actor legitimately is). The
recurring cost: `validate_cfg` normally populates several config fields as
a side effect, and every field it would have set becomes a bare
`ConfigAttributeError`/`KeyError`/`TypeError` the first time RLinf's own
code reads it — never caught by import-only or unit-test verification,
only by a real launch. Known instances, so a future adapter's config audit
starts from this list instead of zero:

- `runner.weight_sync_interval`, `runner.per_worker_log_path` (drop the
  `distributed_log_dir=` kwarg to `Cluster(...)` entirely rather than
  reading this field — matches every adapter's actual entry-script fix).
- `env.train.rollout_epoch` (and `env.eval.rollout_epoch` if eval is ever
  enabled) — defaults to `1`.
- `actor.model.model_type`/`policy_setup` — `EnvWorker.env_interact_step`
  calls `prepare_actions(..., model_type=..., policy=model_cfg.get(
  "policy_setup", None), ...)` unconditionally regardless of env type; for
  ManiSkill, `prepare_actions_for_maniskill`'s `if "panda" in policy:`
  passthrough branch is what a custom, already-ManiSkill-ready action
  policy wants — `policy_setup` unset is `"panda" in None`, a `TypeError`,
  not silence.
- `actor.model.is_lora`, `actor.model.add_value_head` — bare reads inside
  `get_model()`/FSDP construction (Pattern B only).
- `actor.global_batch_size % (actor.micro_batch_size * actor_world_size)
  == 0` and `env.*.max_steps_per_rollout_epoch % actor.model
  .num_action_chunks == 0` — real constraints on chosen values, not
  skippable by avoiding `validate_cfg` (asserted directly in
  `EmbodiedFSDPActor.__init__`/library code, Pattern B only).
- `algorithm.entropy_bonus`/`reward_type`/`logprob_type`/`entropy_type` —
  bare reads in `train_micro_batch` (Pattern B only).
- A full `actor.fsdp_config` block (`amp_autocast`, `grad_scaler`,
  `mixed_precision` sub-keys) even for a hand-rolled minimal config used
  outside the normal Hydra composition — reuse the
  `hybrid_engines/fsdp@actor.fsdp_config` default group rather than
  hand-writing this from scratch (Pattern B only).
- `torch.distributed` must be initialized (`RANK`/`WORLD_SIZE`/
  `LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT`) even to construct
  `FSDPModelManager` standalone, for a single-rank "cluster" — not obvious
  from reading, only from constructing it directly (Pattern B only).

## Design guidance: making future rl-garden work easy to plug into RLinf

This is the forward-looking half of this document. Nothing here is a
mechanism that needs building now — it's how to write new off-policy,
on-policy, and network code so that *if* an RLinf adapter is ever wanted
for it, the adapter is small, following the same shape that made all three
shipped adapters cheap.

### Network / policy design

- **Give any log-prob/entropy-producing method a `sum_dims: bool = True`
  (or equivalent) escape hatch, not just a summed scalar.** RLinf's own
  loss functions want per-action-dimension, unsummed log-probs/entropy;
  rl-garden's own training conventionally wants a summed scalar. Both
  `DiagGaussianActor.action_log_prob`/`evaluate_action_log_prob` and
  `PPOPolicy.forward`/`evaluate_actions` gained this flag non-breakingly
  (default preserves existing behavior everywhere else) specifically to
  satisfy RLinf's model protocol without a second code path. Any new
  actor distribution class should have this from the start.
- **Provide a single-pass `act_with_value_and_logprob(obs, state=None) ->
  (actions, values, log_prob, entropy, updated_state)`-shaped method on
  any on-policy policy**, even if `state` is always `None`/unused today.
  This is what lets a rollout wrapper fill `prev_values`/`prev_logprobs`
  in one forward pass with no algorithm-specific code, and the `state`
  parameter (a no-op today) is exactly the seam a future stateful
  (recurrent/transformer) policy would use to thread hidden state through
  `Trajectory.forward_inputs`, without changing the call signature every
  other policy on the same contract already uses.
- **Keep a policy's value/critic head reachable as its own clearly-named
  submodule**, not buried inside a shared trunk with no distinguishing
  name. RLinf's own FSDP optimizer construction (and any future adapter
  that wraps a policy for FSDP) buckets parameters into separate LR groups
  by substring-matching a name like `"value_head"` — a policy whose critic
  parameters are impossible to name-match generically needs bespoke
  adapter code where one wouldn't otherwise be needed.
- **Never assume an incoming observation/action tensor is already on the
  right device inside a `forward()`-adjacent method meant to be called
  from outside the current process's own rollout loop.** Add an explicit
  device move (a `self._device`/`next(self.parameters()).device` property
  is enough) at the boundary. Cheap to add now, expensive to debug later
  as a cross-process `RuntimeError`.

### Off-policy algorithm design

- **Keep the hook contract that already worked (SACCore) load-bearing for
  any new off-policy algorithm**: `_sample_train_batch(batch_size)`,
  `_critic_loss(data) -> (loss, info)`, `_actor_loss(obs) -> (loss,
  log_prob)`, callable without the algorithm itself owning optimizer
  stepping. A new algorithm that overrides something *outside* this hook
  set (like `DDPG`'s inline `train()` with no `_sample_train_batch` at
  all) is not a branch to add to the existing adapter — it needs its own
  adapter, and that's fine; don't force a shared adapter to special-case
  a structurally different algorithm via `hasattr` branching.
- **Keep replay buffers swappable as objects, not just sample-able via a
  method an adapter could patch.** Any algorithm that mixes multiple data
  sources (like RLPD's online/offline mix) should keep that mixing logic
  in `_sample_train_batch` reading from `self.replay_buffer`, not baked
  into a method an external caller might paper over — this is what let
  the RLPD adapter swap in RLinf's buffer with zero special-casing and
  zero risk of silently dropping the offline half.
- **A new off-policy algorithm's construction should only ever need
  `single_observation_space`/`single_action_space`/`num_envs` from its
  `env=` argument** — never call `.reset()`/`.step()` outside `learn()`'s
  own rollout loop. This is what makes `OfflineEnvSpec`-style no-live-env
  construction (already required for every RLinf adapter) free, rather
  than something each new algorithm has to re-earn.

### On-policy algorithm design

- **Factor any recurrence-style computation (GAE, N-step returns, any
  future advantage estimator) as a standalone pure function operating on
  plain tensors, separate from whatever stateful buffer class normally
  calls it.** `rl_garden/buffers/gae.py::compute_gae` is the concrete
  precedent: extracted out of `RolloutBuffer.compute_returns_and_advantage`
  specifically so an RLinf adapter could reshape RLinf's own batch layout
  into the function's plain-tensor inputs and reuse the exact same,
  already-tested math — without needing a whole `RolloutBuffer` object or
  duplicating the recurrence. Any future on-policy addition (a different
  advantage estimator, a value-clipping variant) should default to this
  shape from the start.
- **Keep rollout collection and the gradient-update step separable at the
  call-site level**, the way `OnPolicyAlgorithm.learn()` already collects
  a full rollout via `rollout_buffer.add(...)` calls before ever touching
  `self.train()`. An RLinf adapter needs to drive these independently
  (rollout happens in a separate worker process from training); an
  algorithm whose update logic is entangled with its own rollout loop
  can't be adapted without first separating them.
- **If a future on-policy algorithm is genuinely sequence-aware (recurrent
  or transformer-based), keep hidden state as an explicit, opaque value
  threaded as a function argument/return** (as `RecurrentPPOPolicy
  .act_recurrent` already does), never as implicit module state mutated
  in place. That's what makes `Trajectory.forward_inputs` a viable carrier
  for it later — an adapter can thread an opaque object through a
  dict value without knowing its internal shape, but only if the
  algorithm's own code already treats it as an explicit value rather than
  a hidden side effect.

### If an algorithm is ever meant to run under RLinf's *own* FSDP loop (Pattern B), not just Pattern A

This is a stronger, more specific set of constraints than the general
guidance above — only relevant if you specifically want RLinf's own
training loop and multi-GPU sharding, not just its orchestration:

- The network wrapper's `forward()` must be defined explicitly, never left
  to resolve via multiple-inheritance dispatch from a mixin.
- Every invocation of the wrapped model from adapter code — not just
  RLinf's own internal calls — must go through `model(...)` (`__call__`),
  never a bare method call on the model object, with no exceptions once
  any sharding beyond `no_shard` is in play.
- Any parameter needing its own optimizer/LR group must be reachable at a
  `named_parameters()` path containing the exact substring RLinf's
  optimizer builder matches on, and — if that name is only reachable via
  an alias to an already-registered submodule — that alias must be
  assigned *first*, before the path that would otherwise win the
  parameter-name-deduplication race.
- Any process-global registration this wrapper needs from RLinf (a custom
  model-type key, or any other mutation of RLinf's own module-level
  state) must happen at import time of a module every relevant Ray worker
  process actually imports — never only from a driver-process entry
  point.

## What's not built

Real, verified-as-viable-in-principle gaps, listed for completeness — none
scheduled, none blocking anything shipped:

| Gap | Status |
|---|---|
| DDPG contract (TD3, DrQv2) | Structurally separate hook set from SACCore (no `_sample_train_batch`, incompatible `_critic_loss` signature, no `_actor_loss` at all) — would need its own Pattern-A adapter, not a branch on the existing one. |
| RLPDHybrid discrete-critic head | Its `_train_discrete_critic` path runs entirely outside `_critic_loss`/`_actor_loss` — a SACCore-contract adapter driving training only through those two hooks would silently never train the discrete gripper head. Needs adapter extension or explicit exclusion, not silent partial support. |
| Cal-QL under RLinf | Needs its own MC-returns data pipeline (computed from true episode boundaries, not free from the offline adapter), and RLinf's rank-local replay buffers mean any offline/online mixing-ratio parity claim would need re-validation regardless — see the next row. |
| Rank-local replay/dataset shards | An RLinf design property, not an rl-garden integration gap — N ranks means N disjoint data pools, no amount of rl-garden-side refactoring changes this. Fine for plain SAC; changes the effective data distribution for anything mixing multiple sources (RLPD, Cal-QL) across ranks. |
| SequencePPO (RecurrentPPO, TransformerPPO) | The carrier (`Trajectory.forward_inputs`) and the policy-side seam (`act_with_value_and_logprob`'s `state` parameter, `RecurrentPPOPolicy.act_recurrent`'s explicit hidden-state threading) both already exist and are verified as mechanisms — just not wired into an adapter. |
| Async decoupled PPO (`AsyncPPOEmbodiedRunner`, `decoupled_actor_critic`) | A real, structurally different runner RLinf ships; the shipped PPO adapter targets the sync `EmbodiedRunner` instead. |
| Multi-node placement, eval (`val_check_interval`), checkpoint-resume | Every adapter's verification so far is single-node, eval-disabled, no resume tested. |
| TDMPC2 / TDMPC2Multitask | No RLinf worker or runner shape hosts world-model learning combined with planning-based action selection — no host to adapt into, refactored or not. |

All RLinf integration code lives under `rl_garden/integrations/rlinf/`.
