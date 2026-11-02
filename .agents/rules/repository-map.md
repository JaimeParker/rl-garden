# Repository Map

Top-level layout of `rl-garden`. Read `AGENTS.md` first for the project-level
agent rules; this file is a plain orientation reference, not a rules doc.

- `rl_garden/algorithms/` — online, offline, and off-to-online algorithms.
  `observation_encoders.py` holds `ObservationEncoderMixin` (Layer C: turns an
  algorithm's `encoder_config`/`obs_groups`/`critic_encoder_config`/
  `encoder_sharing` into built feature extractors via `_policy_extractor_kwargs()`).
  `model_based.py` holds `ModelBasedAlgorithm` (`OffPolicyAlgorithm` subclass;
  TD-MPC2/DreamerV3's online base) and `offline_model_based.py` holds
  `OfflineModelBasedAlgorithm` (`OfflineRLAlgorithm` subclass; offline base);
  both share the three-hook gradient step in `model_based_step.py`
  (`ModelBasedStepMixin`) — see `.agents/rules/adding-algorithm.md`'s
  "model-based algorithms" section. Flat modules `tdmpc2.py` (`TDMPC2Core` +
  `TDMPC2`)/`tdmpc2_multitask.py`/`dreamer_v3.py`, no sub-package.
  Also flat: `pwm.py` (`PWMCore`/`PWM`, SHAC-style actor-critic on a frozen
  multitask world model; offline host subclass), `rssm_pretrain.py`
  (`RSSMPretrain`, reward-free RSSM pretraining; offline host subclass),
  `ncrl.py` (`NCRL(DreamerV3)`, online fine-tuning with rehearsal and BC
  guidance), and `supermimic.py` (`SuperMimicCore`/`SuperMimic`, online
  BPTT actor through an explicit-state `StateSpaceWorldModel` with a
  lambda-return critic on the imagined trajectory; `SuperMimicPolicy`).
- `rl_garden/observations/` — the observation composition contract (Layer A):
  `ObservationConfig` ("what is observed" — state/rgb/depth cameras/extra_state
  auxiliary keys, image size, frame stacking), `ObservationSchema`/`Modality`/
  `ObsEntry` (derived from an observation space), `ObsGroups`/`resolve_obs_groups`
  (asymmetric actor/critic observation keys), and the strict `state`/`state_<name>`/
  `rgb_<cam>`/`depth_<cam>` key-vocabulary validation every backend/dataset loader
  must satisfy.
- `rl_garden/policies/` — policy composition and actor/critic modules.
  `BasePolicy` owns the extractor contract: policies subclass it, call
  `super().__init__(..., actor_extractor=..., critic_extractor=...,
  encoder_sharing=...)`, and use `extract_actor_features()`/`extract_critic_features()`
  and their dimension properties for head construction.
  `graph_ppo_policy.py` holds `GraphPPOPolicy`, the `PPOPolicy` whose trunk is a
  `GraphSpec` (built by `PPO` when `policy_graph` is set).
  `bc_guide.py` holds the behaviour-cloning guide policy NCRL swaps in during
  guidance windows; `action_mask.py` holds the padded-action-dim masking (`masked_bounded_normal`)
  for NCRL's multi-embodiment actor.
- `rl_garden/buffers/` — tensor, dict, Monte-Carlo, and rollout buffers.
  `sequence_replay_buffer.py` holds `SequenceReplayBuffer`, the unified
  `(T, N)` windowed buffer strict single-episode windows (TD-MPC2,
  `cross_episode=False`) and boundary-tolerant windows
  (`RecurrentReplayBuffer`/`TransformerReplayBuffer`, `cross_episode=True`)
  are both built from.
  `state_sequence_replay_buffer.py` holds `StateSequenceReplayBuffer`
  (strict `SequenceReplayBuffer` plus `sample_history`: in-episode
  observation history with first-frame replication, SuperMimic's
  imagination starts).
  `rehearsal_sequence_buffer.py` holds `RehearsalSequenceBuffer` (online + retrieved
  prior episodes mixed per batch, NCRL); `sequence_dataset.py` reads the on-disk
  image sequence-dataset format (`MmapSequenceDataset`, used by `rssm_pretrain`).
- `rl_garden/encoders/` — state, CNN, RGBD/proprio, pooling, FiLM, and ResNet
  encoders (Layer B — "how observations are encoded"). `config.py` holds
  `EncoderConfig` (the `--encoder.*` CLI surface); `registry.py` holds
  `ENCODER_REGISTRY`/`EncoderSpec` (backbone name -> factory); `factory.py`
  holds `build_observation_encoder(observation_space, encoder_config, ...)`,
  the one entry point that builds a `FlattenExtractor` (state-only) or
  `CombinedExtractor` (Dict with images) feature extractor.
- `rl_garden/networks/` — actor, critic, value, and MLP backbone builders,
  plus model-based-RL numeric primitives shared across world-model
  implementations (`symlog.py`, `twohot.py`, `normed_mlp.py`,
  `running_scale.py`, `running_norm.py` (`RunningNorm`, exact-merge running
  mean/var buffers), `q_ensemble.py`, `returns.py`'s `lambda_return`,
  `init.py`).
  `networks/graph/` holds the YAML policy-graph machinery for the PPO family:
  `spec.py` (`GraphSpec`, validation, canonical form, the YAML reference in its
  module docstring), `registry.py` (node-type registry, `rlgarden.graph_nodes`
  entry points), `nodes.py` (in-tree nodes and the module builder), `losses.py`
  (auxiliary-loss registry and info-target checks), `executor.py` (flat, step
  and sequence runs, hidden-state layout) and `builtin.py` (the graphs that
  reproduce `PPOPolicy`/`RecurrentPPOPolicy`). User guide:
  [`docs/guides/policy-graphs.md`](../../docs/guides/policy-graphs.md).
- `rl_garden/world_models/` — the abstract `WorldModel`/`State` contract
  (`base.py`) every model-based algorithm's learned dynamics model
  implements, the `SequenceBatch` window batch every `model_loss` reads
  (`batch.py`), the shared `imagine()` rollout primitive (`imagine.py`),
  TD-MPC2's `LatentConsistencyModel` (`latent_consistency.py`) and
  `MultitaskWorldModel` (`multitask_latent_consistency.py`),
  `BoundTaskWorldModel` (`bound_task.py`, a multitask model with one task id
  bound, usable by `imagine()`/the planner), and DreamerV3's `RSSM`
  (`rssm.py`) with its `ObservationDecoder` (`decoder.py`), and
  `StateSpaceWorldModel` (`state_space.py`, an explicit-simulator-state model
  with pluggable `StateCodec`/`AnalyticMDP` and learned-or-analytic heads). A world
  model owns encoder/dynamics/reward/(optional termination) only — actor
  and critic live on the owning `BasePolicy` (see `rl_garden/policies/`
  below and `.agents/rules/adding-algorithm.md`'s "model-based algorithms"
  section).
- `rl_garden/planners/` — decision-time planners (TD-MPC2's CEM/MPPI,
  `mppi.py`) that search over a `WorldModel` rollout instead of the policy
  producing an action directly. Not shared with imagination-based
  algorithms (DreamerV3), which use `rl_garden.world_models.imagine`
  instead.
- `rl_garden/common/` — logging, shared CLI arguments, environment arguments,
  checkpoint I/O, optimization, types, and utilities.
- `rl_garden/envs/` — backend registry and implementations, environment factories,
  wrappers, and custom environments.
  `envs/dm_control/` is the dm_control suite wrapper package (`config.py`,
  `env.py`, `tasks.py`, `custom_tasks/` for NCRL's custom tasks), registered by
  `envs/backends/dm_control.py`; `envs/wrappers/multitask_pad.py` pads
  observations/actions of one task to a multitask model's dims (PWM evaluation).
- `rl_garden/models/` — ACT and reward models.
- `rl_garden/training/` — registry base and independent `online/`, `offline/`, and
  `off2on/` packages; each holds `args.py`, `registry.py`, `runner.py`, and an `entrypoints/`
  package with one module per algorithm (all imported by the registry, followed by the
  modules of the `rlgarden.algorithms` entry-point group). `evaluation.py` is checkpoint
  evaluation (`rl-garden-eval`; `examples/eval_checkpoint.py` wraps it) and `cli.py` the
  console-script mains. Extension contract for algorithms in other packages:
  [`docs/guides/extending.md`](../../docs/guides/extending.md), reference package
  `tests/fixtures/downstream_extension/`.
  New entrypoints: `online/entrypoints/ncrl.py` (`ncrl`), `online/entrypoints/supermimic.py`
  (`supermimic`), `offline/entrypoints/pwm.py` (`pwm`) and
  `offline/entrypoints/rssm_pretrain.py` (`rssm_pretrain`); the latter two are bespoke runners.
- `real_world/` — optional submodule
  ([rlgarden-real-world](https://github.com/JaimeParker/rlgarden-real-world)):
  real-robot actor/learner loops, the `franka_real`/`real_robot` env backends,
  the teleop source layer, and the SERL / HIL-SERL / residual / SiLRI
  training entrypoints. It imports `rl_garden` as a library and registers its
  backends and algorithms through the `rlgarden.env_backends` and
  `rlgarden.algorithms` entry-point groups. Its `robot_infra/` directory is the
  torch-free device layer (`rlgarden-robot-infra`: devices, daemon,
  teleoperation hardware), a second distribution of the same repository with no
  dependency on `rl_garden`.
- `examples/` — thin training dispatchers (wrappers of `rl_garden/training/cli.py`) and
  specialized experiment entrypoints.
- `configs/` — reusable preset configs for training.
- `scripts/` — shell launchers with experiment defaults.
- `tools/` — standalone utilities: `conversion/` (checkpoint/dataset format
  conversion), `diagnostics/` (Q-value and parity probes), `reproductions/`
  (third-party baseline reproduction runners that *patch* a specific idea into
  a copied source tree, e.g. `run_iql_fixed_mixing.py` — different purpose
  from `baselines/`, see below).
  World-model tools in `conversion/`: `convert_ncrl_dataset.py` (NCRL npz
  episodes to the sequence dataset), `build_ncrl_retrieval_db.py` (first-frame
  embedding DB for rehearsal), `convert_tdmpc2_checkpoint.py` (upstream/HF TD-MPC2
  multitask checkpoint to an rl-garden `MultitaskWorldModel`).
- `baselines/` — top-level package for running *unmodified* official JAX
  baseline repos (Cal-QL, wsrl, IQL-jax) against rl-garden's canonical
  environments, for pure numeric comparison. See
  [`.agents/runbooks/baseline-install.md`](.agents/runbooks/baseline-install.md).
  Not to be confused with `rl_garden/integrations/rlinf/`, which is the
  opposite direction — rl-garden's own algorithms running as workers under
  RLinf.
- `pretrained/` — externally pretrained weights (ResNet, ACT), outside the
  importable package tree.
- `tests/` — unit tests and accelerator/backend integration smoke tests.
- `docs/` — public documentation, split into `guides/` (operational how-to),
  `design/` (architecture and rationale), and `roadmaps/` (migration-tracking
  notes).
  World-model docs: `guides/{ncrl,supermimic,dm-control-integration}.md`.
- `3rd_party/` — vendored reference submodules and external clones; read-only,
  do not edit unless explicitly requested. `Cal-QL`, `wsrl`, and
  `implicit_q_learning` are registered as real git submodules (see
  `baselines/baselines.yaml`); the rest are untracked reference clones.

## Module Naming Convention

- Module names are plain and descriptive. There are no underscore-prefixed
  modules in `rl_garden/`: a private helper lives next to its single user, and
  a helper used by several modules lives in `common/` under a plain name.
  Underscore prefixes belong to functions, classes, and attributes inside a
  module, not to file names.
- The only `pkgutil`-scanned directories are `rl_garden/training/<phase>/entrypoints/`
  (mirrored by `rl_garden_real_world/training/entrypoints/` in the real-world
  repo) and `rl_garden/envs/backends/`. Every module in those directories is
  imported by its registry, so put only registering modules there; the
  registries skip `_`-prefixed names as a safety valve, not as a way to
  place infrastructure next to entry modules.
- Phase infrastructure (`args.py`, `registry.py`, `runner.py`, `demo_loading.py`
  in the real-world mirror) sits one level above `entrypoints/`, where discovery
  never sees it.
