# HIL-SERL Roadmap

This document tracks what HIL-SERL (`3rd_party/hil-serl/`) capability was
**not** migrated in the initial `hil_serl` training-method landing (see
`rl_garden/training/real_world/hil_serl.py`), and why. HIL-SERL itself splits
into three largely independent training scripts -- `train_rlpd.py` (online
RLPD + demo mixing + HITL + reward classifier), `train_bc.py` (standalone
BC), and `train_hgdagger.py` (BC pretrain + iterative correction against a
growing on-disk dataset) -- and this round targeted only the
`train_rlpd.py` capability set.

Items 3 and 7 below are now landed (marked inline) -- they were judged the
two blocking gaps for a usable end-to-end real-world human-in-the-loop run
(no way to produce a reward-classifier checkpoint; no crash recovery for
long-running online data collection). Items 8, 9, and 10 later landed as
well. The rest remain open.

A follow-up source-level comparison against `3rd_party/hil-serl/examples/
train_rlpd.py` (the only authoritative HIL-SERL implementation) surfaced one
concrete bug in the just-landed item 7, plus three additional gaps not
previously tracked here -- see "Item 7 correction" and items 8-10 below (11
remains open).

A second, broader comparison round split HIL-SERL's own code-structure table
(`3rd_party/hil-serl/README.md`'s "Overview and Code Structure") into its 7
leaf categories -- `examples`, `serl_launcher.{agents,data,vision,wrappers}`,
`serl_robot_infra.{robot_servers,franka_env}` -- and diffed each against the
current `rl-garden` implementation. It re-confirmed items 1-11 below are
still accurate, sharpened several of them (see "Item 8 correction" and the
caveats added to items 2, 5, 9, 10, and 11), and surfaced a substantial batch
of previously untracked findings: one concrete correctness bug (item 12), a
set of missing capabilities (items 13-20), and a class of "implemented, but
the default/behavior differs from what HIL-SERL's own real-robot recipes
actually use" divergences, tracked separately in "Behavioral Divergences
from HIL-SERL's Real-Robot Defaults" below.

### Item 7 correction: `--offline_dataset_path` silently discarded by `init_demo_buffer`

`DemoInterventionMixin.init_demo_buffer()` (`rl_garden/buffers/
demo_intervention.py`) unconditionally replaces `self.offline_replay_buffer`
with a fresh empty buffer. `_run_learner` (`rl_garden/training/real_world/
hil_serl.py`) calls `build_rlpd_hybrid(args, ...)` (which loads
`args.offline_dataset_path` into `offline_replay_buffer` if set) and *then*
`agent.init_demo_buffer(...)` -- so a user who passes `--offline_dataset_path`
to `hil_serl` doesn't get a "can't use both" error, they get their loaded
data **silently discarded** with no diagnostic. Item 7's original text framed
this as an unsupported combination; it's actually a data-loss bug. (Now fixed
-- see item 8 below.)

### 8. Demo-data preloading (`--demo_path` equivalent) -- LANDED

HIL-SERL's own `--demo_path` (`train_rlpd.py:458-466`) is not RLPD's generic
`--offline_dataset_path` (a structured maniskill_h5/minari dataset loader) --
it's a list of raw pickled transition dicts, unconditionally loaded into the
same `demo_buffer` that intervention data grows into, on every process start
*including restarts* (loaded again each time, then crash-recovery snapshots
are glob-reloaded on top). Before this round, `rl-garden`'s demo buffer had
no preloading path at all -- `init_demo_buffer()` always started empty, so a
`hil_serl` run could only ever bootstrap from human intervention accumulated
during that run, with no way to seed it from prior teleop demonstrations (the
way real HIL-SERL runs almost always start).

Now built:

- `HilSerlLearnerLoop.__init__` gained a `demo_dataset_paths: Sequence[str]`
  parameter (glob patterns), loaded via `_load_demo_datasets()` into
  `agent.offline_replay_buffer` (through `add_demo_transition`) unconditionally
  on every start, before crash-recovery snapshot reload -- mirrors HIL-SERL's
  own always-reload-`demo_path` behavior exactly.
- File format matches the existing buffer-snapshot pkl convention (a pickled
  list of `{"obs", "next_obs", "action", "reward", "done", ...}` CPU-tensor
  dicts -- the same shape `HilSerlLearnerLoop._snapshot()` writes), not
  HIL-SERL's own numpy `{"observations", "actions", ...}` dict shape. This is
  a deliberate format choice, not a fidelity gap: it means a previous run's
  `checkpoint_dir/demo_buffer/*.pkl` snapshot files can be reused directly as
  another run's `--demo_dataset_paths` seed data.
- New `HilSerlArgs` field: `demo_dataset_paths: Sequence[str] = ()`.
- `DemoInterventionMixin.init_demo_buffer()` now raises if
  `offline_replay_buffer` is already populated (e.g. from
  `--offline_dataset_path`) instead of silently overwriting it -- turns the
  item-7-correction bug above into an explicit error. The underlying
  limitation (RLPD's static offline-dataset slot and HIL-SERL's demo-buffer
  slot can't both be used on one `RLPDHybrid` instance) is unchanged, just no
  longer silent.

**Item 8 correction**: `_load_demo_datasets()` (called from
`HilSerlLearnerLoop.__init__`, before `_reload_snapshots()`) increments
`self._received` for every preloaded demo transition, even though those only
reach `agent.add_demo_transition` (the demo buffer), never
`agent.replay_buffer` (the online buffer). `received_transitions` gates
`LearnerLoop.run()`'s wait-for-`learning_starts` check
(`rl_garden/real_world/learner_loop.py`). HIL-SERL's own analogous gate
(`while len(replay_buffer) < config.training_starts`, `train_rlpd.py:278`)
checks only the online buffer's length -- its `demo_buffer` (populated from
`--demo_path`) never counts toward it. Concretely: a `hil_serl` run seeded
with `--demo_dataset_paths` supplying >= `learning_starts` transitions can
start training with **zero** real online transitions collected, which
HIL-SERL never allows (it always waits for genuine online data regardless of
demo/prior volume). `tests/test_real_world_hil_serl_loops.py`'s
`test_hil_serl_learner_loop_preloads_demo_dataset_paths` currently asserts
`received_transitions == 3` after preloading with zero live transitions --
the test encodes this behavior rather than catching it. Not yet fixed.

**Also missing**: HIL-SERL's `examples/record_demos.py` (plain teleop
trajectory recording, no success/failure labels, producing exactly the
`--demo_path` pkl shape) has no `rl-garden` equivalent -- only
`rl_garden/models/reward/success/collect_data.py` exists, and it's scoped to
success/failure-labeled reward-classifier data, not general demo seeding.
Anyone wanting to seed `--demo_dataset_paths` today has to write ad hoc
recording code themselves.

### 9. Real-world agent/policy checkpointing -- LANDED

`LearnerLoop.run()` (`rl_garden/real_world/learner_loop.py`) used to drive
training via `agent.train(gradient_steps)` directly and never call
`OffPolicyAlgorithm._maybe_save_periodic_checkpoint`/`_save_checkpoint` --
those are only invoked from `OffPolicyAlgorithm.learn()`'s single-process
rollout loop, which no real-world entrypoint (`serl` or `hil_serl`) uses, and
that method also gates on `_global_step` (rollout-step count), which
`LearnerLoop` never advances. HIL-SERL's own learner checkpoints full agent
state (weights + optimizer) every `checkpoint_period` *gradient-update* steps
and resumes via `checkpoints.restore_checkpoint`
(`train_rlpd.py:249-355,419-429`).

Turned out most of the plumbing already existed: `build_rlpd`/
`build_rlpd_hybrid` (`rl_garden/training/online/{rlpd,rlpd_hybrid}.py`)
already call `agent.load(args.load_checkpoint, ...)` when `--load_checkpoint`
is set, and both `serl.py`/`hil_serl.py`'s `_run_learner` already pass
`args`/`checkpoint_dir` straight through -- so **resume via
`--load_checkpoint` already worked** for real-world learners, just untested
and undocumented. The only actual gap was the periodic-*save* trigger.

Now built, in the base `LearnerLoop` (shared by `serl` and `hil_serl`, not
`HilSerlLearnerLoop`-only -- both bypass `learn()` identically):

- `BaseAlgorithm.global_update` -- new read-only property exposing
  `_global_update` (restored correctly by `load_state_dict`, so it stays
  monotonic across a resume, unlike a loop-local counter).
- `LearnerLoop._maybe_save_periodic_checkpoint()`, called after every
  `_train_step()`: if `agent.checkpoint_dir` and `agent.checkpoint_freq > 0`,
  saves `checkpoint_{global_update}.pt` once the gradient-update count
  crosses each `checkpoint_freq` multiple -- keyed on gradient updates,
  matching HIL-SERL's own `step` semantics in `learner()`, not
  received-transition count.
  `include_replay_buffer=agent.save_replay_buffer` (existing flag, respected
  as-is).
- `LearnerLoop.run()`'s `finally` block now also does a one-shot `final.pt`
  save when `agent.save_final_checkpoint` is set -- covers clean stops
  (`stop()`, `KeyboardInterrupt`) in addition to the periodic saves that are
  the actual crash-recovery mechanism.
- No new `HilSerlArgs`/`SerlArgs` fields needed -- `checkpoint_dir`/
  `checkpoint_freq`/`load_checkpoint`/`save_replay_buffer`/
  `save_final_checkpoint` were already exposed via the shared `CheckpointArgs`
  (`rl_garden/common/cli_args.py`) and already flowed into the agent.

Known architectural consequence: `tests/test_real_world_learner_loop.py`'s
documented "LearnerLoop only touches this minimal `OffPolicyAlgorithm`
surface" contract grew to include `checkpoint_dir`/`checkpoint_freq`/
`save_replay_buffer`/`save_final_checkpoint`/`global_update`/`save(...)` --
an intentional, documented expansion (both `_FakeAgent` and the file's
docstring were updated), not scope creep.

**Caveat**: the cadence match to HIL-SERL claimed above is not exact.
HIL-SERL's `checkpoint_period` counts `learner()`'s outer loop iterations
(`train_rlpd.py:314-355`), each of which bundles `config.cta_ratio` (default
2) total `agent.update()` calls -- one critic-only update plus one combined
critic+actor+temperature update. `rl-garden`'s `_global_update` increments
once per raw gradient step inside `sac_core.py`, and `checkpoint_freq` gates
on that finer-grained count. The same numeric `--checkpoint_freq` value
therefore triggers a save at a different real cadence on the two sides
(roughly 2x apart under HIL-SERL's default `cta_ratio=2`) -- not a bug,
since both are internally consistent and the flag is `rl-garden`'s own to
define, but worth knowing when comparing configured values across the two
codebases.

### 10. Real-world evaluation mode -- LANDED

HIL-SERL's actor supports `--eval_checkpoint_step`/`--eval_n_trajs`: load a
specific saved checkpoint and run a fixed number of real-robot episodes to
measure success rate, no data collection or training (`train_rlpd.py:71-110`).

Now built as a third `hil_serl`-only role (kept out of the shared
`RealWorldFrankaArgs.role` to keep this round's blast radius limited to
`hil_serl`; `serl.py` is untouched):

- `HilSerlArgs.role: Literal["actor", "learner", "eval"]`, plus
  `eval_n_trajs: int = 10`.
- `_run_eval(args)` (`rl_garden/training/real_world/hil_serl.py`): builds the
  env via the existing `_build_env` (same wrapper stack as actor/learner,
  including the reward classifier), builds the agent via `build_rlpd_hybrid`
  -- requiring `--load_checkpoint` (raises `ValueError` if unset, unlike
  `_run_actor`'s scratch-agent path which discards it) -- and runs
  `eval_n_trajs` episodes with `deterministic=True` fixed (ignores
  `--deterministic_actor`, which governs the training-time actor's
  exploration instead). No sync client, no learner process, no data pushed
  anywhere. Per-episode success is read off `terminated`:
  `RewardClassifierWrapper` sets it `True` exactly when the classifier judges
  success (`rl_garden/envs/wrappers/reward_classifier.py`); a `truncated`
  (timeout) episode counts as a failure. Prints per-episode and aggregate
  success-rate/mean-return.
- `--fwbw` + `role="eval"` raises `NotImplementedError` -- out of scope this
  round, same "single-arm only" boundary already documented elsewhere.

**Caveats found comparing against HIL-SERL's own eval loop**
(`train_rlpd.py:67-110`):

- HIL-SERL's `--eval_checkpoint_step` mode is actually *stochastic*
  (`agent.sample_actions(..., argmax=False)`, `train_rlpd.py:88-91`), despite
  being labeled "eval." `rl-garden`'s `role="eval"` forces
  `deterministic=True` unconditionally -- stricter than what it's modeled
  on, not a bug, but it means the two sides' measured "success rate" isn't
  measuring the same policy behavior.
- HIL-SERL's eval loop only checks `done` (`while not done:`,
  `train_rlpd.py:86,95-98`), never `truncated` -- a timeout-only episode
  wouldn't end the reference loop. `rl-garden`'s `_run_eval` explicitly
  breaks on `bool(terminated) or bool(truncated)`, counting truncation as a
  failure. Whether HIL-SERL's actual Franka envs also set `done=True` on
  timeout (which would make this moot in practice) was not verified.

### 11. `grasp_penalty` reward-shaping term -- not yet migrated

HIL-SERL's hybrid single/dual-arm SAC (`serl_launcher/agents/continuous/
sac_hybrid_{single,dual}.py`) trains the discrete gripper critic against
`rewards + grasp_penalty` (a shaping term discouraging unnecessary
open/close actuation), sourced from `info["grasp_penalty"]` on the actor
side. `RLPDHybrid._train_discrete_critic` (`rl_garden/algorithms/
rlpd_hybrid.py`) trains against the raw environment reward only -- `info`
never carries a `grasp_penalty` key anywhere in the `rl-garden` stack. This
is an algorithm-fidelity gap (affects gripper-policy behavior), not an
infrastructure gap.

**Precision note**: `grasp_penalty` is not produced anywhere in the agent --
it's produced by an *env wrapper*, `GripperPenaltyWrapper`/
`DualGripperPenaltyWrapper` (`3rd_party/hil-serl/serl_robot_infra/
franka_env/envs/wrappers.py:339-397`), a `gym.RewardWrapper` that penalizes
unnecessary gripper open/close actuation and stashes the penalty in
`info["grasp_penalty"]`; the agent-side critic loss then reads it off the
batch. Both pieces are missing in `rl-garden`: no such wrapper exists in
`rl_garden/envs/wrappers/`, and `RLPDHybrid._train_discrete_critic` has no
parameter for it.

## Not Yet Migrated

### 1. HG-DAgger training mode + the growing-dataset buffer layer

`train_hgdagger.py`'s iterative correction loop needs a continuously
growing, periodically-reloaded on-disk demo/correction dataset -- a
fundamentally different data path from `LearnerLoop`'s current model (live
online transitions, or a static offline dataset loaded once via RLPD's
`offline_dataset_path`). This was flagged early and deliberately scoped out
because it isn't needed for `train_rlpd.py`-style HIL-SERL.

- `LearnerLoop._refresh_offline_data()` (`rl_garden/real_world/learner_loop.py`)
  already exists as a no-op hook reserved for exactly this -- both `serl`
  and `hil_serl` currently leave it unoverridden.
- Reference design (from `3rd_party/RLinf`'s real-world DAgger/SFT path,
  researched during this migration): a disk-file dataset pipeline fully
  decoupled from the in-memory RL replay buffer -- a LeRobot-format writer,
  a separate offline ETL step, and a lazily-reloaded PyTorch `Dataset` that
  picks up newly written data. Don't try to force growing demo data through
  the existing in-memory buffer interface.

### 2. BC (behavior cloning) agent

`3rd_party/hil-serl/serl_launcher/serl_launcher/agents/continuous/bc.py`'s
`BCAgent` -- reuses the same actor network as SAC (`Policy` class) with a
pure NLL loss, no critic. Confirmed via reading HIL-SERL's own code that BC
is **never** combined with the online RLPD loss at training time (no BC
regularization term, no weight-loading from a BC checkpoint into the SAC
actor) -- it's used standalone, either for `train_bc.py`'s own pretraining
runs or as the bootstrap policy for `train_hgdagger.py`. Since it's fully
independent of everything landed so far, it can be added later as its own
`rl_garden/algorithms/` addition without touching `RLPDHybrid`,
`hil_serl.py`, or any of the loop/sync base classes.

**Note**: `rl_garden/algorithms/bc.py`'s `BCPolicy` hardwires a
tanh-squashed Gaussian actor (`rl_garden/networks/actor_critic.py:93,184-186`
via `SquashedGaussianActor`). HIL-SERL's own `BCAgent.create()` defaults to
`tanh_squash_distribution=False` (`bc.py:143-145`) -- unsquashed by default.
Only matters if/when BC gets wired into real-world training; currently both
implementations are fully standalone, so this has no live effect.

### 3. Reward classifier training script + data-collection convention -- LANDED

Originally, `rl_garden/envs/franka_real/classifier.py` only had the model
class (`SuccessClassifier`) and the inference-side loader
(`load_classifier_fn`) -- enough for `RewardClassifierWrapper` to consume an
already-trained checkpoint, but nothing to produce one.

This is now built, and the whole reward-model surface moved to
`rl_garden/models/reward/` (a pre-existing, previously-unimported directory
of offline HDF5-labeled classifiers -- `classifiers/{base,color,alignment,hsv}`
-- reorganized so the directory's semantics genuinely match "reward model",
not just "classifier"; future kinds like VLM- or rule-based reward models are
expected to land as further siblings). `success/` is the new sibling for the
online real-robot success classifier:

- `rl_garden/models/reward/success/model.py` -- `SuccessClassifier`/
  `load_classifier_fn`, moved here unchanged from `franka_real/classifier.py`.
- `rl_garden/models/reward/success/data.py` -- `SuccessClassifierDataset`,
  reading `{"obs": obs}` pickle lists split by success/failure glob pattern
  (mirrors HIL-SERL's `classifier_data/*_{success,failure}_*.pkl` convention).
- `rl_garden/models/reward/success/train.py` -- standalone argparse CLI (not
  routed through `BaseAlgorithmRegistry`; supervised classifier training
  doesn't fit the RL Args/env_backend shape), mirroring HIL-SERL's own
  `examples/train_reward_classifier.py`: balanced success/failure batch
  sampling, `binary_cross_entropy_with_logits`, `RandomShiftsAug`
  (`rl_garden/encoders/augment.py`, reused as-is) for augmentation, no
  train/val split, fixed epoch count.
- `rl_garden/models/reward/success/collect_data.py` -- ported HIL-SERL's
  `examples/record_success_fail.py` mechanism: `pynput` global spacebar
  listener marks the most recently stepped transition a success, everything
  else defaults to failure. Camera capture is caller-supplied (see
  `env.py`'s module docstring -- this was already true of the whole
  `franka_real` stack before this round: `envs/backends/franka_real.py`
  always constructs the env with `camera_capture=None`), so the CLI
  entrypoint has no camera wired up; call `main(camera_capture=...)` from a
  cell-specific script to actually record images.

While fixing `classifiers/hsv/__init__.py`'s pre-existing broken import
(`rl_garden.reward_models...` -> `rl_garden.models.reward...`, which made
the whole `classifiers/hsv` submodule unimportable), two more latent
import-time bugs surfaced in the same pre-existing `classifiers/base/` code
and were fixed as part of making the directory importable at all:
`metrics.py` eagerly imported `sklearn` (not a declared dependency anywhere
in `pyproject.toml`) and `transforms.py` eagerly imported `torchvision`
(also undeclared) -- both moved to lazy, function-local imports.
`classifiers/hsv/generate_labels.py` also had two leading module docstrings,
which is a `SyntaxError` for the `from __future__ import annotations` line
immediately after them; reduced to one.

### 4. Dual-arm hybrid gripper (`sac_hybrid_dual`)

`RLPDHybrid`/`DiscreteCritic` only implement HIL-SERL's `sac_hybrid_single`
shape (one continuous arm, a 3-way discrete gripper Q-head). HIL-SERL's
`sac_hybrid_dual` (two arms, 12D continuous action, gripper choices fused
into one joint 9-way discrete Q-head via the Cartesian product of both
grippers) was not implemented -- rl-garden's `FrankaRealEnv` only supports a
single Franka arm today, so there's no env to drive a dual-arm policy
against yet. This is the same "single-arm only" boundary already documented
for the rest of the real-world RL stack.

### 5. Franka bridge server endpoints: `/set_load`, `/close_gripper_slow`, `/move_gripper`, `/update_param`

Comparing `3rd_party/hil-serl/serl_robot_infra/robot_servers/franka_server.py`
against the base SERL version already ported into
`robot_infra/controller/real/franka_bridge.py` found these HIL-SERL routes
that weren't carried over:

- `/set_load` (POST) -- calls the ROS service `/franka_control/set_load`
  with `mass`/`F_x_center_load`/`load_inertia`, presumably for updating the
  arm's dynamics model after a tool/gripper change. Requires a new ROS
  service dependency (`franka_msgs.srv.SetLoad`) not currently wired up.
- `/close_gripper_slow` (POST) -- a slower gripper-close variant
  (Robotiq-specific, see below).
- `/move_gripper` (POST, `franka_server.py:339-345`) -- continuous gripper
  position control, backed by `move()` on both `franka_gripper_server.py`
  and `robotiq_gripper_server.py`. `rl-garden`'s gripper server only exposes
  binary open/close, so no policy action space depending on a continuous
  gripper width can be driven through this bridge today.
- `/update_param` (POST, `franka_server.py:223-225,378-381`) -- runtime
  dynamic-reconfigure of the cartesian impedance controller's
  stiffness/damping via `ReconfClient`. No equivalent in `franka_bridge.py`
  -- impedance parameters can't be tuned at runtime over HTTP.

Not a gap: the many fine-grained getter routes HIL-SERL exposes (`/getpos`,
`/getvel`, `/getforce`, `/getq`, `/getdq`, `/getjacobian`, etc.) are
deliberately consolidated into `franka_bridge.py`'s single `/getstate`
response instead of being ported 1:1 -- see that file's own docstring, which
scopes it to only what `FrankaBridgeClient` actually calls. `/getstate`'s
response is missing `q`/`dq`/`jacobian` specifically (nothing downstream
currently reads them), which makes `rl_garden/envs/franka_real/
bridge_client.py`'s docstring -- which claims those fields are present but
unused -- inaccurate; worth a doc fix independent of any behavior change.

### 6. Robotiq gripper support

HIL-SERL's `robotiq_gripper_server.py` differs from the Franka-hand gripper
server already ported (inverted `gripper_pos` reporting, a partial-open
default position instead of fully-open, a lower default close force, plus
the `close_slow()` method backing endpoint 5 above). Out of scope: v1's
Franka bridge (`robot_infra/controller/real/gripper_server.py`) only
supports the Franka hand, matching the SERL v1 design's original scoping
decision (Robotiq flagged there as "a future addition if needed").

### 7. Online/demo buffer split + periodic disk snapshot (crash recovery) -- LANDED

`3rd_party/hil-serl/examples/train_rlpd.py:140-233` tracks two separate
buffers on the actor side: every transition goes into `data_store` (all
online transitions), and only the transitions collected while
`info["intervene_action"]` is present also go into `intvn_data_store` (the
"demo" buffer -- a continuously growing set of human-corrected transitions,
not a static pre-collected dataset). Both are in-memory and are what the
learner actually samples from at train time. Separately, every
`config.buffer_period` steps, the transitions accumulated since the last
period are pickled to `checkpoint_path/buffer/transitions_{step}.pkl` and
`checkpoint_path/demo_buffer/transitions_{step}.pkl` -- a crash-recovery
snapshot, reloaded by globbing both directories on restart
(`train_rlpd.py:457-490`), not the live sampling source.

Both pieces are now built, reusing existing machinery rather than adding a
third mixing mechanism:

- **Demo buffer**: `rl_garden/buffers/demo_intervention.py`'s
  `DemoInterventionMixin(PriorDataReplayMixin)` adds `init_demo_buffer()`/
  `add_demo_transition()`, populating the *same* `offline_replay_buffer`/
  `offline_data_ratio` slot `PriorDataReplayMixin` already uses for RLPD's
  static `--offline_dataset_path` prior dataset -- just grown incrementally
  instead of loaded once. `_sample_train_batch`/`_concat_replay_samples`/
  `_shuffle_batch` are all inherited unchanged. **Known limitation**:
  `--offline_dataset_path` and HIL-SERL's live demo buffer share this one
  slot, so this round doesn't support using both at once on the same
  `RLPDHybrid` instance -- see "Item 7 correction" above: this now raises
  instead of silently discarding the loaded offline dataset.
  `RLPDHybrid(DemoInterventionMixin, RLPD)` in
  `rl_garden/algorithms/rlpd_hybrid.py`.
- **Intervention tagging**: `ActorLoop`/`FWBWActorLoop`
  (`rl_garden/real_world/actor_loop.py`) gained an
  `_extra_transition_fields(info) -> dict` hook (no-op default), overridden
  by `HilSerlActorLoop` to tag every pushed transition with
  `{"intervened": "intervene_action" in info}`. **Known limitation**: FWBW
  mode's `_run_actor` drives the generic, un-subclassed `FWBWActorLoop`
  (not `HilSerlActorLoop`), so FWBW-mode hil_serl runs don't currently tag
  intervened transitions -- everything goes to the online buffer.
- **Routing + snapshot**: `HilSerlLearnerLoop`
  (`rl_garden/real_world/hil_serl/learner_loop.py`) overrides
  `_on_transition` to route the way HIL-SERL's own actor loop does
  (`train_rlpd.py:178-193`, corrected -- see below): *every* transition
  (popped `intervened` flag) goes into `agent.replay_buffer`, and intervened
  ones are *additionally* copied into `agent.add_demo_transition` -- the demo
  buffer is a duplicated subset of the online buffer, not a disjoint
  partition. Pickles the accumulated-since-last-snapshot transitions
  (pre-device-transfer, CPU tensors, for portability) to
  `<checkpoint_dir>/buffer/transitions_{n}.pkl` (every transition) /
  `<checkpoint_dir>/demo_buffer/transitions_{n}.pkl` (intervened only) every
  `--buffer_period` received transitions, glob-reloading both directories on
  startup (received-count is tallied off the "buffer" reload only, since
  "demo_buffer" entries are duplicates of ones already in "buffer" -- reloading
  both without this would double-count intervened transitions). This is still
  the *only* buffer persistence path for `hil_serl` training --
  `LearnerLoop.run()` never triggers the existing
  `save_replay_buffer`/`include_replay_buffer` checkpoint mechanism
  (`rl_garden/algorithms/off_policy.py:56,323`), so there's no double-reload
  risk between the two.

  **Correction**: the initial landing routed exclusively (intervened ->
  demo buffer *only*, non-intervened -> online buffer *only*), which is not
  what HIL-SERL does -- its actor inserts every transition into `data_store`
  unconditionally and *additionally* inserts intervened ones into
  `intvn_data_store` (`train_rlpd.py:178-193`). Under the old exclusive
  routing, `rl-garden`'s online buffer never contained any human-corrected
  data at all, changing the online-half sampling distribution relative to
  HIL-SERL (where intervened transitions are present in both the online and
  demo halves of the mix). Fixed to match, scoped entirely to
  `HilSerlLearnerLoop` (no change to `DemoInterventionMixin`, `ActorLoop`, or
  base `LearnerLoop`).
- New `HilSerlArgs` fields: `demo_buffer_size`, `demo_data_ratio`,
  `buffer_period`.

### 12. `gripper_pos` normalization bug + other `robot_servers` deviations

`robot_infra/controller/real/gripper_server.py` was ported from HIL-SERL's
`franka_gripper_server.py` but diverges from it in ways not previously
caught:

- **Missing `/0.08` normalization** (the actual bug): HIL-SERL normalizes
  `gripper_pos` to roughly `[0, 1]` -- `self.gripper_pos =
  np.sum(msg.position) / 0.08` (`franka_gripper_server.py:64-66`).
  `gripper_server.py`'s equivalent line computes `float(sum(msg.position))`
  with no division, despite an adjacent comment claiming it "matches SERL's
  own convention." The resulting value is ~12.5x HIL-SERL's scale, and it
  feeds directly into `FrankaRealEnv`'s state observation
  (`rl_garden/envs/franka_real/env.py`) -- any policy trained against this
  observation is seeing a different numeric range than HIL-SERL's own
  policies did for the same physical gripper opening. Not yet fixed.
- **No open/close idempotence guard**: HIL-SERL tracks
  `self.binary_gripper_pose` and short-circuits a repeated `/open_gripper`
  or `/close_gripper` call if the gripper is already in that state
  (`franka_gripper_server.py:21-43`). `gripper_server.py` has no such guard
  -- every call unconditionally republishes a new
  `MoveActionGoal`/`GraspActionGoal`, sending redundant grasp commands every
  control step a policy holds the gripper closed/open.
- **`load_gripper` roslaunch arg not passed explicitly**: HIL-SERL's
  `franka_server.py` passes `load_gripper:=true/false` to `impedance.launch`/
  `joint.launch` based on gripper type. `franka_bridge.py` omits this arg
  entirely -- currently benign since `3rd_party/serl_franka_controllers`'s
  launch files default it to `true` (the correct value, since `rl-garden`
  only supports the Franka hand today), but implicit rather than explicit.

### 13. Safety box only clamps position, not orientation

HIL-SERL's `FrankaEnv.clip_safety_box()`
(`3rd_party/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py:185-207`)
clips both `xyz_bounding_box` and `rpy_bounding_box` (with special-cased
wraparound handling for the first Euler angle). `FrankaRealEnv.step()`
(`rl_garden/envs/franka_real/env.py`) only clips `target_pos` against
`safety_low`/`safety_high`, and `FrankaRealEnvConfig.safety_box_low/high`
(`config.py`) is a 3-tuple (position only) -- there is no software limit on
end-effector orientation. A runaway policy action or aggressive human teleop
input can rotate the end-effector to an arbitrary orientation with nothing
to stop it.

### 14. No quaternion-to-Euler/rotvec conversion on the observation side

HIL-SERL applies `Quat2EulerWrapper`/`Quat2R2Wrapper`
(`3rd_party/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py:103-144`)
to convert `tcp_pose`'s quaternion component into Euler (or 6D
rotation-matrix) form before it reaches the policy's observation space.
`FrankaRealEnv._obs_from_state()` concatenates the raw quaternion directly
into the flat state vector -- no such conversion wrapper exists anywhere in
`rl_garden/envs/wrappers/`. The action side already uses rotvec deltas
correctly (matching HIL-SERL's action convention); the observation side has
not been brought in line.

### 15. Missing `franka_env` wrapper variants: `HumanClassifierWrapper`, `MultiStageBinaryRewardClassifierWrapper`, `GripperCloseEnv`

`RewardClassifierWrapper` (`rl_garden/envs/wrappers/reward_classifier.py`,
item 3 above) only covers HIL-SERL's simplest case,
`MultiCameraBinaryRewardClassifierWrapper` (single-stage, model-driven
success). Two other wrapper variants HIL-SERL provides
(`3rd_party/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py`) have no
`rl-garden` equivalent:

- `HumanClassifierWrapper` (`wrappers.py:15-34`) -- blocks on `input()` at
  episode end for a human to type success/failure, an alternative to the
  learned classifier for tasks where training one isn't worth it yet.
- `MultiStageBinaryRewardClassifierWrapper` (`wrappers.py:69-100`) --
  sequential multi-stage success tracking (per-stage `received` bookkeeping,
  summed reward, episode ends only once every stage is satisfied) -- needed
  for tasks with multiple sequential sub-goals.

Also missing: `GripperCloseEnv` (`wrappers.py:180-204`), an `ActionWrapper`
that truncates the action space to 6D (drops the gripper dimension,
zero-padding it back) for gripper-always-closed tasks -- no equivalent
action wrapper exists in `rl_garden/envs/wrappers/`.

### 16. Pico and SpaceMouse intervention detection are structurally different (informational)

`robot_infra/teleop/spacemouse/spacemouse_teleop_wrapper.py`'s
`SpaceMouseTeleOpWrapper` closely mirrors HIL-SERL's own
`SpacemouseIntervention` (`3rd_party/hil-serl/serl_robot_infra/franka_env/
envs/wrappers.py:207-252`): always-on rate control, intervention decided by
a pure magnitude deadzone (`norm(expert_a) > threshold`). `rl-garden`'s
other teleop device, `robot_infra/teleop/utils/telo_op_control_twist.py`'s
`EETwistTeleOpWrapper` ("pico"), is *bind-gated* instead -- it only reports
a nonzero (and therefore "intervened") twist while a physical bind button is
held, computing a delta-since-bind, and falls back to a zero/not-intervened
sample whenever no fresh device packet has arrived. Both converge on the
same `TeleOpSample.intervened` contract consumed by
`TeleopInterventionWrapper`, but the underlying activation semantics differ:
physically moving a Pico controller without pressing its bind button never
triggers intervention, whereas any SpaceMouse motion above the deadzone
always will. Not a bug -- just worth knowing which device gives which
behavior when choosing `--teleop_device`.

### 17. No real camera capture integration

HIL-SERL provides a full camera integration module,
`3rd_party/hil-serl/serl_robot_infra/franka_env/camera/{video_capture,
rs_capture,multi_video_capture}.py` -- a threaded `VideoCapture` wrapping
`RSCapture` (direct `pyrealsense2` RealSense SDK integration), wired into
`FrankaEnv.init_cameras()`/`get_im()`. `rl-garden` has no equivalent module
anywhere. `rl_garden/envs/backends/franka_real.py`'s `make_train_env`/
`make_eval_env` never pass `camera_capture`, so it defaults to `None`
throughout the real-world stack (already noted in item 3's text as a
pre-existing condition of the whole `franka_real` backend, not something
this migration introduced) -- meaning no `hil_serl` run today has actual
camera images without a cell-specific integration written by hand.

### 18. No real-world rollout video recording

HIL-SERL's `VideoRecorder`/`VideoWrapper`
(`3rd_party/hil-serl/serl_launcher/serl_launcher/wrappers/{video_recorder,
video_wrapper}.py`) record rollout video to disk for debugging. `rl-garden`
has no equivalent anywhere in the real-world path (the only video-writing
code in the repo, `rl_garden/envs/robotwin/adapter.py`, is RoboTwin
simulation eval logging, unrelated). There is currently no way to record
video of a real-robot `hil_serl`/`serl` rollout for after-the-fact
debugging.

### 19. Replay-buffer scalability gaps: no image-frame dedup, no bounded actor-transport queue

Two `serl_launcher.data` mechanisms that don't affect correctness but do
affect resource usage at scale have no `rl-garden` equivalent:

- **`pack_obs_and_next_obs` frame dedup**: HIL-SERL's
  `MemoryEfficientReplayBuffer` (`3rd_party/hil-serl/serl_launcher/
  serl_launcher/data/memory_efficient_replay_buffer.py`) stores each pixel
  frame once per timestep and reconstructs stacked `obs`/`next_obs` via a
  sliding window at sample time. `rl_garden/buffers/dict_buffer.py`'s
  `DictReplayBuffer` allocates and writes fully separate `obs`/`next_obs`
  arrays on every `add()` -- full duplication per image key. A real
  memory-usage difference for long real-world sessions with image
  observations, not a correctness issue.
- **Bounded actor-side transport queue**: HIL-SERL's actor pushes into a
  `QueuedDataStore(50000)` (`train_rlpd.py:507-508`), giving implicit
  backpressure if the learner falls behind. `rl_garden/real_world/sync.py`'s
  `ActorSyncClient` uses an unbounded `queue.Queue()` -- if the learner is
  slow or down, the actor-side queue grows without limit instead of
  applying HIL-SERL's implicit bound/backpressure.

### 20. Temporal observation stacking not wired into the real-world path

HIL-SERL's `ChunkingWrapper`
(`3rd_party/hil-serl/serl_launcher/serl_launcher/wrappers/chunking.py`)
stacks *all* observation keys (state + images) into a temporal window and is
environment-agnostic -- applied uniformly to sim and real robot envs alike,
plus supports receding-horizon action-chunk execution. `rl-garden`'s closest
analog, `ImageFrameStackWrapper`
(`rl_garden/envs/wrappers/frame_stack.py`), only stacks image keys (state
stays single-frame) and has no action-chunk concept, and more importantly is
only wired into the ManiSkill simulation backend
(`rl_garden/envs/maniskill/env.py`) -- `hil_serl`'s `_build_env`
(`rl_garden/training/real_world/hil_serl.py`) never applies it. Real Franka
rollouts currently get no temporal observation stacking at all.

## Behavioral Divergences from HIL-SERL's Real-Robot Defaults

These are all cases where `rl-garden` *has* the capability HIL-SERL uses,
but the default configuration -- or, in one case, an unconditional
implementation choice -- differs from what HIL-SERL's own real-robot
recipes (`serl_launcher/serl_launcher/utils/launcher.py`,
`examples/experiments/*/config.py`) actually run. None of these are missing
features; all are one flag or one default value away from parity, except
the first, which is hardcoded on the `rl-garden` side.

- **`backup_entropy` is hardcoded `True`, not configurable**: `SAC.__init__`
  (`rl_garden/algorithms/sac_core.py`) has no `backup_entropy` parameter --
  target Q always subtracts `alpha * logpi`. HIL-SERL's real-robot configs
  hardcode `backup_entropy=False`
  (`serl_launcher/serl_launcher/utils/launcher.py:83,133`). Unlike the other
  items below, this can't currently be matched by passing a different flag.
- **Actor-side LayerNorm defaults off**: `RLPDHybridArgs` overrides
  `critic_use_layer_norm=True` but not `actor_use_layer_norm` (inherits
  `False`, `rl_garden/training/online/_args.py`). HIL-SERL's
  `make_sac_pixel_agent_hybrid_single_arm` sets `use_layer_norm=True` on
  both critic and policy kwargs (`launcher.py:120-130`).
- **Image augmentation defaults off for the real-world entrypoint**:
  `VisionArgs.image_augmentation` defaults to `"none"`
  (`rl_garden/common/cli_args.py`), and `RLPDHybridArgs` doesn't override it
  the way `DrQv2TrainingArgs` does (to `"random_shift"`). HIL-SERL
  unconditionally wires DrQ-style random-crop augmentation into every
  real-robot agent build (`launcher.py:190`). Pass
  `--image_augmentation random_shift` to match.
- **ResNet backbone defaults unfrozen, from-scratch**: HIL-SERL's real-robot
  configs (all using `encoder_type="resnet-pretrained"`) unconditionally
  freeze the backbone (`stop_gradient` in `resnet_v1.py`'s `pre_pooling`),
  training only the pooling head. `rl-garden`'s `ResNetEncoder` only freezes
  when the caller passes both `pretrained_weights` *and*
  `freeze_resnet_encoder`/`freeze_resnet_backbone=True`; `VisionArgs`'
  defaults leave all three off, so the real-world path trains the whole
  ResNet from scratch by default.
- **Default pooling method differs**: HIL-SERL's real-robot configs hardcode
  `pooling_method="spatial_learned_embeddings"` (`sac.py:468,487`).
  `ResNetEncoder` defaults to `"spatial_softmax"`
  (`rl_garden/encoders/resnet.py`), and nothing in the real-world entrypoint
  overrides it. Both mechanisms exist on the `rl-garden` side
  (`rl_garden/encoders/pooling.py`) -- this is a default-value mismatch, not
  a missing feature.
- **REDQ ensemble defaults reflect the RLPD paper, not HIL-SERL's real
  practice**: `RLPDHybridArgs` defaults to `n_critics=10,
  critic_subsample_size=2` (matching `RLPD`'s own docstring). HIL-SERL's
  actual real-robot launcher calls for every agent type (RLPD, hybrid
  single, hybrid dual) hardcode `critic_ensemble_size=2,
  critic_subsample_size=None` (`launcher.py:86-87,136-137,186-187`) -- no
  REDQ subsampling was used on real robots in practice.
- **Discrete (gripper) critic always stop-gradients the shared encoder**:
  `RLPDHybrid._train_discrete_critic` calls `extract_features(...,
  stop_gradient=True)` unconditionally
  (`rl_garden/algorithms/rlpd_hybrid.py`). HIL-SERL's `grasp_critic_loss_fn`
  backpropagates into the grasp critic's own pooling head
  (`sac_hybrid_single.py:233-289`, `grad_params=params`) -- only the
  pretrained backbone convolutions are frozen, not the whole encoder path
  from the discrete critic's perspective.
- **FiLM conditioning is unused on both sides** (not a gap): HIL-SERL's
  `FilmConditioning` layer is only reachable through `-film` ResNet config
  variants that none of its real-robot agent constructors ever select.
  `rl_garden/encoders/film.py` is likewise an explicit, documented stub not
  wired into any algorithm. Noted here so it isn't mistakenly re-flagged as
  a missing feature later.
