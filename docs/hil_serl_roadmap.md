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
long-running online data collection). The rest remain open.

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

### 5. Franka bridge server endpoints: `/set_load`, `/close_gripper_slow`

Comparing `3rd_party/hil-serl/serl_robot_infra/robot_servers/franka_server.py`
against the base SERL version already ported into
`robot_infra/controller/real/franka_bridge.py` found two additions HIL-SERL
made that weren't carried over:

- `/set_load` (POST) -- calls the ROS service `/franka_control/set_load`
  with `mass`/`F_x_center_load`/`load_inertia`, presumably for updating the
  arm's dynamics model after a tool/gripper change. Requires a new ROS
  service dependency (`franka_msgs.srv.SetLoad`) not currently wired up.
- `/close_gripper_slow` (POST) -- a slower gripper-close variant
  (Robotiq-specific, see below).

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
  `RLPDHybrid` instance. `RLPDHybrid(DemoInterventionMixin, RLPD)` in
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
  `_on_transition` to route by the `intervened` flag (popped before storage)
  into `agent.replay_buffer` or `agent.add_demo_transition`, and pickles the
  accumulated-since-last-snapshot transitions (pre-device-transfer, CPU
  tensors, for portability) to `<checkpoint_dir>/buffer/transitions_{n}.pkl`
  / `<checkpoint_dir>/demo_buffer/transitions_{n}.pkl` every
  `--buffer_period` received transitions, glob-reloading both directories on
  startup. This is still the *only* buffer persistence path for `hil_serl`
  training -- `LearnerLoop.run()` never triggers the existing
  `save_replay_buffer`/`include_replay_buffer` checkpoint mechanism
  (`rl_garden/algorithms/off_policy.py:56,323`), so there's no double-reload
  risk between the two.
- New `HilSerlArgs` fields: `demo_buffer_size`, `demo_data_ratio`,
  `buffer_period`.
