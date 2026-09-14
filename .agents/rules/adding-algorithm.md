# Adding a New Algorithm

This document is the authoritative guide for adding a new RL/IL algorithm to
rl-garden. Read it before touching `rl_garden/algorithms/`, any
`rl_garden/training/{online,offline,off2on}/` file, or any `_args.py`.

## Overview

An algorithm lives in two places:

```
rl_garden/algorithms/my_algo.py                 ← learning logic (networks, update rule)
rl_garden/training/<phase>/my_algo.py           ← CLI wiring, env/dataset construction, registration
```

The training file is auto-discovered by the phase registry (`discover()` imports
every non-`_`-prefixed module in the package); no other file needs to change to
get a working `--algorithm my_algo` CLI, beyond the checklist in part 6.

| Phase | Runner | Registry package | Example file |
|---|---|---|---|
| online | `run_online(args, *, obs_tag, make_env_request, build_agent, post_learn=None)` | `rl_garden/training/online/` | `online/sac.py` |
| offline | `run_offline(args, *, build_agent)` | `rl_garden/training/offline/` | `offline/iql.py` |
| off2on | `run_off2on(args, *, build_agent, algorithm)` | `rl_garden/training/off2on/` | `off2on/iql.py` |

Algorithm families, by base class:

| Family | Base class(es) | Examples |
|---|---|---|
| Online RL (rollout + replay/buffer) | `OffPolicyAlgorithm` / `OnPolicyAlgorithm` | SAC, PPO, TD3 |
| Model-based online RL (world model + planner/policy) | `ModelBasedAlgorithm(OffPolicyAlgorithm)` | TD-MPC2 |
| Offline RL with a replay buffer | `OfflineRLAlgorithm` (+ `run_offline`) | IQL, CQL, AWAC |
| Off2on (offline pretrain, then online) | `<Algo>Core` / `<Algo>(Core, OfflineRLAlgorithm)` / `_<Algo>RolloutTrainingShell(Off2OnReplayMixin, Core, OffPolicyAlgorithm)` / `Off2On<Algo>(Shell)` | IQL, AWAC (canonical) |
| Chunked-dataset imitation (no replay buffer) | `OfflineRLAlgorithm`, own `train()` | BC, DiffusionBC, FlowBC, DAgger |

There is no `rl_garden/algorithms/components/` package; do not invent one.

---

## Part A — Algorithm implementation (`rl_garden/algorithms/`)

### 1. Choose a base class

| Base class | Use for | Abstract methods | Constructor env arg |
|---|---|---|---|
| `OffPolicyAlgorithm` | SAC, TD3, and variants | `_setup_model()`, `train(gradient_steps, compute_info=False) -> dict[str, float]` | live vectorized env |
| `ModelBasedAlgorithm(OffPolicyAlgorithm)` | TD-MPC2 and variants (world model + planner/amortized policy) | `_setup_model()`, `_update_model(batch) -> (metrics, posterior)`, `_update_actor_critic(batch, posterior) -> metrics`, `_update_targets()` | live vectorized env |
| `OnPolicyAlgorithm` | PPO and variants | `_setup_model()`, `train() -> dict[str, float]` | live vectorized env |
| `OfflineRLAlgorithm` | IQL, CQL, AWAC, and the chunked-dataset imitation family | `train(gradient_steps, compute_info=False) -> dict[str, float]` | `OfflineEnvSpec` (spaces + `num_envs`, no `reset`/`step`) |
| `BaseAlgorithm` | custom loops fitting none of the above | `_setup_model()`, `learn(total_timesteps)` | `(env, eval_env=None, seed=1, device="auto", logger=None)` |

`learn()` is already implemented on `OffPolicyAlgorithm` (GPU rollout + replay
loop), `OnPolicyAlgorithm` (rollout buffer + GAE), and `OfflineRLAlgorithm`
(`learn()` → `learn_offline()` → module-level `run_offline_pretraining()` in
`offline.py`) — only direct `BaseAlgorithm` subclasses implement `learn()`
themselves.

### 2. Minimal off-policy skeleton

```python
# rl_garden/algorithms/my_algo.py
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm

class MyAlgo(OffPolicyAlgorithm):
    def __init__(self, env, eval_env=None, *, my_lr: float = 3e-4, **kwargs):
        super().__init__(env, eval_env, **kwargs)
        self.my_lr = my_lr
        self._setup_model()

    def _setup_model(self) -> None:
        # Build networks, optimizers, replay buffer. Everything on self.device.
        self.policy = ...
        self.my_optimizer = ...

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        batch = self.replay_buffer.sample(self.batch_size)
        ...
        return {"train/loss": loss.item()}

    def _optimizer_names(self) -> tuple[str, ...]:
        return ("my_optimizer",)
```

### 3. Minimal chunked-dataset imitation skeleton

For the no-replay-buffer imitation family (`BC`, `DiffusionBC`, `FlowBC`,
`MeanFlowBC`, `A2ABC`, `ConsistencyDistillBC`, `DAgger`):
same class shape as `MyAlgo` above (`__init__` → `_setup_model()` → `train()`,
`_optimizer_names()`), but on `OfflineRLAlgorithm` with an `OfflineEnvSpec`
(exposes only `single_observation_space`/`single_action_space`/`num_envs`, built
from dataset-inferred spaces via
`infer_specs_from_h5`/`infer_box_specs_from_h5`), and the dataset loads once
into dense tensors instead of a `replay_buffer`:

```python
# rl_garden/algorithms/my_bc.py
from rl_garden.algorithms.offline import OfflineEnvSpec, OfflineRLAlgorithm
from rl_garden.buffers.chunked_dataset import load_h5_dataset_as_chunks

class MyBC(OfflineRLAlgorithm):
    _compatible_checkpoint_algorithms = ("MyBC",)

    def __init__(self, env: OfflineEnvSpec, dataset_path: str, *, batch_size: int = 128, **kwargs):
        super().__init__(env=env, batch_size=batch_size, eval_env=None, **kwargs)
        self.dataset_path = dataset_path
        self._setup_model()
        self._load_dataset()  # sets self._obs_history, self._action_chunks, self._dataset_size

    def _load_dataset(self) -> None:
        self._obs_history, self._action_chunks = load_h5_dataset_as_chunks(
            self.dataset_path, horizon_steps=4, cond_steps=1, device=self.device,
        )
        self._dataset_size = self._action_chunks.shape[0]

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        idx = torch.randint(0, self._dataset_size, (self.batch_size,), device=self.device)
        ...  # index self._obs_history/self._action_chunks with idx
        return {"train/loss": loss.item()}
```

### 4. Checkpoint hooks

`save()`/`load()` are implemented once on `BaseAlgorithm` — never reimplement
them; extend only through these hooks:

| Hook | Default | Override for |
|---|---|---|
| `_optimizer_names()` | `("q_optimizer", "actor_optimizer", "alpha_optimizer", "cql_alpha_optimizer")` (`OnPolicyAlgorithm` → `("policy_optimizer",)`) | naming your optimizer attributes |
| `_checkpoint_metadata()` | `{"seed", "device"}` | hyperparameters to validate a resume; always `{**super()._checkpoint_metadata(), ...}` |
| `_extra_checkpoint_state()` / `_load_extra_checkpoint_state()` | `{}` / no-op | LR scheduler state, EMA weights, etc. |
| `_training_state_dict()` / `_load_training_state_dict()` | `{}` / no-op | loop-level counters |
| `_checkpoint_includes_replay_buffer()` | `False` | `True` if a real replay buffer should persist |
| `_ddp_extra_broadcast_modules()` | `[]` | state outside `self.policy` DDP must broadcast (e.g. `SAC`'s `alpha_tuner`) |
| class attr `_compatible_checkpoint_algorithms` | `()` | resuming from a differently-named checkpoint (e.g. `OfflineSAC` from `SAC`) |

`_checkpoint_metadata()` feeds `checkpoint["metadata"]["hyperparameters"]`.
On-disk layout (`checkpoint.py`, `checkpoint_dict()`): `metadata` =
`{algorithm_class, global_step, global_update, observation_space, action_space,
hyperparameters, replay_buffer_path}`; `state` = `{policy, optimizers,
global_step, global_update, training_state, extra}` (the last three from
`_optimizer_names()`/`_training_state_dict()`/`_extra_checkpoint_state()`).

All network tensors and replay buffers must stay on `self.device`; CPU-backed
env observations are moved by `_obs_to_policy_device()` before inference — no
ad-hoc `.cuda()` calls. Avoid NumPy in rollout/replay/update hot paths.

### 5. Observation encoding and policy contract

Every algorithm is observation-type agnostic: it never branches on
`isinstance(obs_space, spaces.Box/Dict)`. Observation space is always
`spaces.Dict` with keys `state`/`state_<name>`/`rgb_<cam>`/`depth_<cam>`
(`rl_garden.observations`); a state-only Box is normalized to
`Dict({"state": Box})` before an algorithm ever sees it.

**Policy contract** (`rl_garden/policies/base.py`):

A policy subclass inherits from `BasePolicy` and calls its `__init__` first
with the observation and action spaces plus extractor(s):

```python
class MyPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, *,
                 actor_extractor: BaseFeaturesExtractor,
                 critic_extractor: BaseFeaturesExtractor | None = None,
                 encoder_sharing: EncoderSharing = "shared_critic_grad",
                 ...):
        super().__init__(
            observation_space, action_space,
            actor_extractor=actor_extractor,
            critic_extractor=critic_extractor,
            encoder_sharing=encoder_sharing,
        )
        # Now build heads with extracted features
```

`critic_extractor=None` means the critic reads through `actor_extractor`
(both `"shared_critic_grad"` and `"shared"` modes); a real `critic_extractor`
means `"separate"`.

Stop-gradient lives **only** in `BasePolicy.extract_actor_features()`, which
detaches iff `encoder_sharing == "shared_critic_grad"` and the two extractors
are the same object. Every policy consumer picks a role:

- **Actor heads** (policy nets, BC/flow/diffusion nets, distill heads,
  teacher/student actors): call `self.extract_actor_features(obs)` and read
  dims from `self.actor_features_dim`.
- **Critic heads** (Q-nets, V-nets, TD-target nets, discriminators,
  reward heads): call `self.extract_critic_features(obs)` and read dims from
  `self.critic_features_dim`. Pass `stop_gradient=True` only for genuinely
  unrelated reasons (frozen-encoder training, eval-only reads); never to
  express the encoder-sharing rule.

**Algorithm setup** (`rl_garden/algorithms/_observation.py`):

`ObservationEncoderMixin` sits on `BaseAlgorithm`, so every algorithm gets it
for free. A concrete algorithm opts in by:

1. Accepting `encoder_config: EncoderConfig | None = None`,
   `obs_groups: ObsGroups | None = None`, `critic_encoder_config: EncoderConfig |
   None = None`, and `encoder_sharing: EncoderSharing | None = None` as
   constructor kwargs, and storing them as `self.encoder_config`/
   `self.obs_groups`/`self.critic_encoder_config`/`self.encoder_sharing`.
   Algorithms without a critic (BC-only) omit `critic_encoder_config` and
   `encoder_sharing`.

2. Calling `self._policy_extractor_kwargs(observation_space,
   augmentation_seed=...)` in `_setup_model()`. This returns a dict of
   `{"actor_extractor", "critic_extractor", "encoder_sharing"}` (or fewer keys
   for BC-only). Pass the dict unpacked into the policy constructor:
   ```python
   extractor_kwargs = self._policy_extractor_kwargs(observation_space)
   self.policy = MyPolicy(..., **extractor_kwargs)
   ```

3. **Required:** declaring a class attribute `encoder_sharing: Literal["shared_critic_grad", "shared", "separate"]`.
   `"shared"` is the class default for BC-only algorithms (no critic) and for IDQL/QGF;
   every other critic-bearing algorithm defaults to `"shared_critic_grad"` (the
   actor path is still stop-gradiented). `"separate"` is needed whenever
   asymmetric `obs_groups.actor != obs_groups.critic` or a distinct
   `critic_encoder_config` is provided by the user. Do NOT declare encoder_sharing
   as a constructor default; instead accept `encoder_sharing: EncoderSharing | None = None`
   and let the resolution logic infer the final value. Optionally restrict allowed
   values by defining a class attribute `encoder_sharing_choices: tuple[Literal[...], ...]`.
   Recurrent/transformer algorithms (RecurrentSAC, RecurrentPPO, TransformerSAC, TransformerPPO)
   must declare `encoder_sharing_choices = ("shared_critic_grad", "shared")` to prevent
   `separate` at preflight. Sharing/asymmetry violations raise `ObservationContractError`
   (a `ValueError` subclass). BC-only algorithms have no encoder-sharing attribute.

**Out-of-contract exceptions** (documented, plain `nn.Module`s):

The following policy classes do not follow the contract above:

- `HILPPolicy`, `FlashSACPolicy` — plain `nn.Module`s, no `BasePolicy`
  inheritance.
- `UniO4MixturePolicy` — plain `nn.Module`, wraps sub-policies.
- `RecedingHorizonPolicy` — inherits `BasePolicy` but does not accept
  `critic_extractor` (always `None`).
- `MultitaskTDMPC2Policy` — no observation/action space or rollout
  `actor_extractor` role to declare (multitask training never touches a live
  env; see `rl_garden/policies/tdmpc2_multitask_policy.py`'s docstring).
  `TDMPC2Policy` (single-task) is now **in** contract (below) and no longer
  belongs on this list.

The FQL family (`FQLPolicy`, `FloQPolicy`, `ValueFlowsPolicy`, `FINOPolicy`)
is an in-contract exception: under `encoder_sharing="separate"`, it keeps an
extra actor-side `actor_bc_flow` encoder built separately by the algorithm
through `FQLCore`; the critic uses `critic_extractor` as normal.

**Model-based algorithms** (TD-MPC2, DreamerV3): the algorithm's
learned dynamics model IS the `actor_extractor` — `TDMPC2Policy`
(`rl_garden/policies/tdmpc2_policy.py`) passes its
`rl_garden.world_models.latent_consistency.LatentConsistencyModel` as
`actor_extractor`, `critic_extractor=None`, `encoder_sharing="shared"`.
`DreamerPolicy` (`rl_garden/policies/dreamer_policy.py`) does the same with
its `rl_garden.world_models.rssm.RSSM`. Both are `WorldModel` subclasses.
`WorldModel` (`rl_garden/world_models/base.py`) does not subclass
`BaseFeaturesExtractor` — its method surface (`encode`/`observe`/`step`/
`reward`/`continue_`/`model_loss`, all operating on a `State =
dict[str, Tensor]`) is much larger than a features extractor's single
encode-and-return-features contract — instead it duck-types the specific
attributes/methods `BasePolicy` actually touches on an extractor
(`features_dim`, `extract()`, `update_normalizer()`; see `WorldModel`'s own
docstring). A world-model-based policy's actor (policy prior / amortized
policy head) and critic (value function) are separate modules owned by the
policy, never by the world model — `rl_garden/networks/q_ensemble.py`,
`rl_garden/networks/running_scale.py`, and TD-MPC2's own actor helpers
(`rl_garden/policies/_tdmpc2_math.py`) are the reusable pieces. TD-MPC2 uses
decision-time planning (CEM/MPPI) injected into the policy; DreamerV3 uses
an amortized actor-critic trained within imagination and needs no planner.

**`ModelBasedAlgorithm`** (`rl_garden/algorithms/model_based.py`,
`OffPolicyAlgorithm` subclass) is the base for this family. Override exactly
three hooks, never `train()`/`_gradient_step()` themselves (both are shared):

- `_update_model(batch) -> (metrics, posterior)` — trains the world model
  (and, for TD-MPC2, the critic in the same backward — a TD-MPC2-specific
  choice, document it if your algorithm does the same). `posterior` is
  whatever LIVE (graph-attached, see `WorldModel.model_loss`'s docstring)
  `State` the world model's own `model_loss()` returned.
  `_update_actor_critic` reads it directly (own gradient path) or detaches it
  first (an already-stepped model, TD-MPC2's case) — that choice belongs to
  `_update_actor_critic`, not here.
- `_update_actor_critic(batch, posterior) -> metrics` — trains the actor (and
  the critic, if not already folded into `_update_model`).
  `TDMPC2._update_actor_critic` detaches `posterior["z"]` before calling
  `_update_pi` for exactly this reason.
- `_update_targets()` — Polyak/hard target-network update(s), run once per
  gradient step after the two above.

`_rollout_action` is decision-time action selection: TD-MPC2's planner
(`self.policy.planner.plan(...)`, threading its OWN `prev_mean`/`t0` state,
kept separate from `self.policy.predict()`'s eval-only state — see
`TDMPC2._rollout_action`'s docstring for why the two must not share state)
or, for an amortized-policy model-based algorithm with nothing decision-time
to plan, the inherited `OffPolicyAlgorithm` default (`self.policy.predict()`)
may be enough — don't override it unless your action selection genuinely
needs algorithm-owned state the policy doesn't already carry.

A `learning_starts`-sized pretrain burst (TD-MPC2's upstream `seed_steps`
semantics) is `_on_learning_starts()` (`OffPolicyAlgorithm` hook, default
no-op) — see that hook's docstring for exactly when it's called and why it
replaces, not adds to, that iteration's regular `train()` call.

**DreamerV3-specific wiring**: uses `SequenceReplayBuffer(cross_episode=True,
priority=False, carry_spec={"deter": ..., "stoch": ...})` to store and restore
the RSSM's recurrent state across training windows. The carry flows through
`OffPolicyAlgorithm`'s `_replay_buffer_add_kwargs` (on rollout) and is written
back via `_replay_buffer_step_kwargs` hooks after gradient steps. The policy
maintains eval-only state through `DreamerPolicy.predict()`, separate from the
buffer's training-time carry.

See `SAC` and `PPO` (`rl_garden/algorithms/sac.py`, `ppo.py`) for the
reference implementations. There is no separate vision-specific wiring path
and no `Vision*` sibling class — a `Dict` observation with
`rgb_<cam>`/`depth_<cam>` keys is handled by the same code path as a
state-only one, driven entirely by `encoder_config`/`obs_groups` and the
schema derived from the observation space.

### 6. Export the class

Add one line to `rl_garden/algorithms/__init__.py` (flat eager import list +
`__all__`): `from rl_garden.algorithms.my_algo import MyAlgo`.

---

## Part B — Online training registration (`rl_garden/training/online/`)

Create `rl_garden/training/online/my_algo.py`. Functions are defined first;
imports plus the `Args` dataclass and `registry.register(...)` go at the
**bottom** of the file, in that order — see `online/sac.py`. Ruff has E402
disabled repo-wide, so no `# noqa: E402` comments are needed there.

```python
# MUST go through construct_agent, not MyAlgo(...) directly -- it records the
# constructor call for --print-config/--dry-run; materialize_config raises if
# no build was recorded, on every run.
def build_my_algo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.my_algo import MyAlgo
    from rl_garden.common.cli_args import resolve_critic_encoder_config
    from rl_garden.training.inspection import construct_agent

    agent = construct_agent(
        MyAlgo, env=env, eval_env=eval_env, my_lr=args.my_lr, seed=args.seed,
        encoder_config=args.encoder if args.obs.is_visual else None,
        obs_groups=args.obs_groups, critic_encoder_config=resolve_critic_encoder_config(args),
        encoder_sharing=args.encoder_sharing,
        logger=logger, checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq, save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint, ...
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent

def run_my_algo(args: "MyAlgoArgs") -> None:
    from rl_garden.common.env_args import make_env_request
    from rl_garden.training.online._runner import run_online
    run_online(args, obs_tag="rgbd" if args.obs.is_visual else "state",
               make_env_request=make_env_request, build_agent=build_my_algo)
               # post_learn=lambda agent: ...  # optional cleanup after learn()

# ---------------------------------------------------------------------------
# Args + registration (bottom of file, no # noqa: E402 -- ruff has it disabled)
# ---------------------------------------------------------------------------
from dataclasses import dataclass

from rl_garden.common.env_args import EnvBackendArgs
from rl_garden.training.online._args import MyAlgoTrainingArgs
from rl_garden.training.online._registry import registry

@dataclass
class MyAlgoArgs(MyAlgoTrainingArgs, EnvBackendArgs):
    """MyAlgo — brief description.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """

def _my_algo_algorithm_cls() -> type:
    from rl_garden.algorithms.my_algo import MyAlgo

    return MyAlgo

registry.register("my_algo", MyAlgoArgs, run_my_algo, algorithm_cls=_my_algo_algorithm_cls)
```

`registry.register` rejects both a duplicate `name` and a duplicate `args_cls` —
reusing another algorithm's `Args` dataclass fails at import time, so give
`MyAlgoArgs` its own class even with no new fields.

### `MyAlgoTrainingArgs` in `_args.py`

Hyperparameters belong in `rl_garden/training/online/_args.py`, not inline in
the run file (unless the algorithm is a one-off). Any algorithm that accepts
observations mixes in `ObservationArgs` (`rl_garden/common/cli_args.py`,
adds `obs`/`encoder`/`obs_groups`/`critic_encoder`/`encoder_sharing`):
`MyAlgoTrainingArgs(EnvRunArgs, CheckpointArgs)` for the algorithm's own
hyperparameters, `VisionMyAlgoTrainingArgs(MyAlgoTrainingArgs,
ObservationArgs)` adding the observation surface (see `VisionSACTrainingArgs`
for the reference). `obs` defaults to state-only, so this one class now
serves both state and visual runs -- unlike the old `VisionArgs`, mixing it
in no longer changes the observation default. Keep the two-layer split only
when the algorithm genuinely needs a different resource budget under either
class regardless of obs choice (e.g. `VisionSACTrainingArgs` overriding
`buffer_size`/`batch_size`/`utd`); otherwise mix `ObservationArgs` straight
into the one `MyAlgoTrainingArgs`. Final composition:
`MyAlgoArgs(VisionMyAlgoTrainingArgs, EnvBackendArgs)`.

---

## Part C — Offline registration (`rl_garden/training/offline/`)

Two shapes. Pick generic unless the algorithm has no replay buffer (the
chunked-dataset imitation family).

### Generic-runner shape (`offline/iql.py`)

Imports and the `Args` dataclass go at the **top** of the file, functions after,
`registry.register(...)` last.

```python
# rl_garden/training/offline/my_algo.py
from dataclasses import dataclass

from rl_garden.training.offline._args import MyAlgoOfflineArgs, OfflineCommonArgs
from rl_garden.training.offline._registry import registry

@dataclass
class MyAlgoArgs(OfflineCommonArgs, MyAlgoOfflineArgs):
    """MyAlgo offline pretraining."""

def build_my_algo(args, env_spec, logger, eval_env=None):
    from rl_garden.algorithms import MyAlgo
    from rl_garden.common.cli_args import resolve_critic_encoder_config
    from rl_garden.training.inspection import construct_agent

    return construct_agent(
        MyAlgo, env=env_spec, my_lr=args.my_lr, seed=args.seed, logger=logger,
        eval_env=eval_env, checkpoint_dir=None, checkpoint_freq=0,
        encoder_config=args.encoder if args.obs.is_visual else None,
        obs_groups=args.obs_groups, critic_encoder_config=resolve_critic_encoder_config(args),
        encoder_sharing=args.encoder_sharing, ...
    )

def run_my_algo(args: MyAlgoArgs) -> None:
    from rl_garden.training.offline._runner import run_offline

    run_offline(args, build_agent=build_my_algo)

def _my_algo_algorithm_cls() -> type:
    from rl_garden.algorithms.my_algo import MyAlgo

    return MyAlgo

registry.register("my_algo", MyAlgoArgs, run_my_algo, algorithm_cls=_my_algo_algorithm_cls)
```

`OfflineCommonArgs` (`offline/_args.py`) already composes `LoggingArgs,
CheckpointArgs, EnvBackendArgs` plus the offline
eval/vision/dataset/replay/optimization mixins — put only algorithm-specific
hyperparameters in `MyAlgoOfflineArgs`. `run_offline(args, *, build_agent)`
calls `build_agent(args, env_spec, logger, eval_env)`; it loads the dataset into
`agent.replay_buffer` via `load_offline_dataset` and, when `args.env_id is not
None and should_create_eval_env(args)`, builds an optional eval env.

### Bespoke-runner shape (`offline/diffusion_bc.py`)

For algorithms with **no replay buffer** — the chunked-dataset imitation family
(`diffusion_bc`, `consistency_distill_bc`, `a2a_bc`,
`hilp`, `opal`, `tdmpc2_multitask`). `run_offline` assumes `agent.replay_buffer`
populated by `load_offline_dataset`, which these algorithms don't have, so the
run file inlines the same config-session and logging setup `run_offline` would
otherwise do, then calls `run_offline_pretraining` directly. Args compose only
`(CheckpointArgs, LoggingArgs)` — no `OfflineCommonArgs`.

```python
# rl_garden/training/offline/my_bc.py — trimmed shape
def _run_my_bc(args, cleanup):
    from rl_garden.algorithms import MyBC, OfflineEnvSpec
    from rl_garden.algorithms.offline import run_offline_pretraining
    from rl_garden.training.inspection import (
        config_session, emit_materialized_config, has_config_session,
        is_dry_run, materialize_config, prepare_standalone,
    )
    from rl_garden.training.offline._registry import registry

    if not has_config_session():
        normalized_args, preflight = prepare_standalone(
            args, registry=registry, training_phase="offline", algorithm="my_bc",
        )
        with config_session(preflight, dry_run=False):
            return _run_my_bc(normalized_args, cleanup)

    # ... build Logger, checkpoint_dir; construct env via OfflineEnvSpec,
    # agent via construct_agent(MyBC, ...) ...
    if is_dry_run():
        emit_materialized_config(env_request=..., env=env, eval_env=None, agent=agent, derived=...)
        return
    materialize_config(env_request=..., env=env, eval_env=None, agent=agent, derived=...)  # then persist_effective_config + logger.update_config

    run_offline_pretraining(
        agent, num_steps=args.num_offline_steps, checkpoint_dir=checkpoint_dir,
        save_filename="my_bc_offline_pretrained.pt", save_replay_buffer=False,
        save_final_checkpoint=args.save_final_checkpoint, log_freq=args.log_freq,
    )

registry.register("my_bc", MyBCArgs, run_my_bc)
```

The `has_config_session()`/`prepare_standalone`/`config_session` recursion and
the dry-run branch are copied from `run_offline` itself
(`rl_garden/common/effective_config.py` has `persist_effective_config`), since
there is no shared runner to call.

### Registry allowlist step (`dataset_path` algorithms only)

`_validate_config` (`algorithm_registry.py`) hardcodes which offline algorithms
require which CLI flag. `tdmpc2_multitask` has its own branch (`--dataset_dir` +
`--mmap_dir`). A separate tuple — `("diffusion_bc",
"consistency_distill_bc", "a2a_bc", "hilp", "opal")` — requires
`--dataset_path`; everything else (the generic-runner shape) requires
`--offline_dataset`. If your new bespoke-runner algorithm uses `--dataset_path`,
add its name to that five-element tuple.

---

## Part D — Off2on registration (`rl_garden/training/off2on/`)

```python
def run_my_algo(args: "MyAlgoOff2OnArgs") -> None:
    from rl_garden.training.off2on._runner import run_off2on
    run_off2on(args, build_agent=build_my_algo, algorithm="my_algo")
```

`run_off2on(args, *, build_agent, algorithm)` calls `build_agent(args, env,
eval_env, logger, checkpoint_dir)`; `algorithm` is passed through for
config/logging and is **not** cross-checked against the registry name.
`Off2OnCommonArgs` (`off2on/_args.py`, composes `EnvRunArgs, CheckpointArgs`)
holds orchestration fields generic across off2on families; compose it with an
algorithm-specific hyperparameter mixin (reusing the offline one where it
exists, e.g. `IQLOff2OnTrainingArgs(Off2OnCommonArgs, OfflineIQLArgs,
OfflineValueArgs)`) and a `Vision...` layer, then `EnvBackendArgs` — see
`off2on/iql.py`.

### The Core / Shell / Off2On class pattern

Off2on = an offline algorithm that also continues training online. Follow
`IQL`/`Off2OnIQL` (and `AWAC`/`Off2OnAWAC`) as the default template — most
algorithms fit this shape with no algorithm-specific overrides:

| # | Class | Role |
|---|---|---|
| 1 | `<Algo>Core` (mixin, no base class) | loss/network/optimizer logic, shared via an explicit `_init_<algo>_params(...)` method (not `__init__`) |
| 2 | `<Algo>(<Algo>Core, OfflineRLAlgorithm)` | the public, pure-offline class |
| 3 | `_<Algo>RolloutTrainingShell(Off2OnReplayMixin, <Algo>Core, OffPolicyAlgorithm)` (internal, do not instantiate) | wires the core into the rollout loop |
| 4 | `Off2On<Algo>(_<Algo>RolloutTrainingShell)` | the public off2on class: constructor + default preset |

`Off2OnReplayMixin` (`rl_garden/algorithms/off2on.py`) already provides every
offline→online transition mechanic that isn't algorithm-specific
(mixed/empty/append replay, adaptive ratio, checkpoint/probe/logging). Leave its
two hooks (`_apply_online_regularizer_override`, `_offline_probe_metrics`) at
their no-op defaults unless your algorithm genuinely needs to change something
at the online switch.

Reach for a non-canonical base only with a real literature-level subtyping
relationship to another algorithm — three known exceptions: Cal-QL's shell
builds on `_CQLRolloutTrainingShell`
(`_CalQLRolloutTrainingShell(Off2OnReplayMixin, CalQLCore,
_CQLRolloutTrainingShell)`) rather than `OffPolicyAlgorithm` directly; SO2's
shell builds on the concrete `SAC` class
(`_SO2RolloutTrainingShell(Off2OnReplayMixin, SO2Core, SAC)`, offline class
`SO2(SO2Core, OfflineSAC)`); `WSRL(_CalQLRolloutTrainingShell)` has no
`WSRLCore` — it builds directly on Cal-QL's shell.

Sanity-check the resulting MRO once — cooperative `super()` calls make
reordering base classes a silent behavior change, not an error:

```bash
python -c "from rl_garden.algorithms import Off2OnMyAlgo; print([c.__name__ for c in Off2OnMyAlgo.__mro__])"
```

---

## Checklist — files a new algorithm touches

1. `rl_garden/algorithms/my_algo.py` — the algorithm class(es), including
   `encoder_sharing` class attribute (required for all critic-bearing algorithms;
   use `"shared_critic_grad"` for most, `"shared"` for IDQL/QGF, restrict to
   `("shared_critic_grad", "shared")` for recurrent/transformer variants).
2. `rl_garden/algorithms/__init__.py` — one import line (+ `__all__` entry).
3. The phase `_args.py` (`training/{online,offline,off2on}/_args.py`) —
   hyperparameter dataclass(es).
4. `rl_garden/training/<phase>/my_algo.py` — the run module (env/dataset
   request, `build_my_algo`, `run_my_algo`, `Args`, `registry.register`).
   **Required:** pass `algorithm_cls=_my_algo_algorithm_cls` to `registry.register()`
   (see Part B/C for the pattern).
5. `rl_garden/training/algorithm_registry.py`'s `_validate_config` allowlist
   tuple — **only** for a bespoke offline algorithm requiring `--dataset_path`
   (see Part C).
6. `tests/test_training_registry.py::test_phase_registries_discover_expected_algorithms`
   — add the new name to the hardcoded set for its phase.
7. A smoke test, `tests/test_<algo>_smoke.py` (see `test_diffusion_bc_smoke.py`,
   `test_sac_smoke.py`), and a CLI test, `tests/test_<algo>_cli.py` (see
   `test_diffusion_bc_cli.py`, `test_off2on_iql_cli.py`).

---

## Verification

```bash
# Registry discovery (same pattern for rl_garden.training.{offline,off2on}._registry):
python -c "
from rl_garden.training.online._registry import registry
registry.discover()
print(sorted(registry.entries()))
"

# Config inspection (no env or agent created). Same two flags work for
# examples/pretrain_offline.py and examples/train_off2on.py in place of
# examples/train_online.py below:
python examples/train_online.py my_algo --print-config 2>/dev/null | python -m json.tool | head -20

# --dry-run exercises the same build_agent/construct_agent path without training:
python examples/train_online.py my_algo --dry-run
```

Run the smallest relevant test set first, then the full suite:

```bash
pytest tests/test_training_registry.py tests/test_<algo>_smoke.py tests/test_<algo>_cli.py -q
pytest tests/ -q
```

Prefer running both on the remote box (see
`.agents/rules/remote-training-sop.md`) over a local dev environment.
