# SuperMimic: Backpropagation Through an Explicit-State World Model

SuperMimic (arXiv 2610.28480) trains a deterministic policy by backpropagating the
return through imagined rollouts of a differentiable world model. The model works on the
simulator's explicit state `s`, not on a learned latent. Each iteration collects one
vectorized env step with the noisy policy; a fractional accumulator schedules updates
according to the official sample reuse ratio. An update has three parts (the paper's Eq. 1-3):

1. **World model (Eq. 1).** An MLP reads normalized state features, the action and the
   per-episode context `xi`, and predicts a normalized local increment. An analytic
   integrator turns the increment into the next state. Training is autoregressive over a
   window of `k` steps: only the first state is real, every later state is the model's own
   prediction, and the loss is the squared error of the increment, expressed in the real
   state's local frame and divided by the running std of the increments, summed over the
   window. Learned reward, observation and termination heads add their own terms.
2. **Actor (Eq. 2).** From replay start states (with their in-episode observation
   history), unroll `H` imagined steps: `a_t = pi(o_hist, g_t)`, `s_{t+1} = f(s_t, a_t)`,
   the next observation comes from the observation head. With soft termination `d_t` and
   survival `c_0 = 1`, `c_{t+1} = c_t (1 - d_{t+1})`, maximize
   `J = sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)`. The world model and critic are
   frozen but the gradient flows through them; there is no action noise in imagination.
3. **Critic (Eq. 3).** On the same (detached) trajectory, regress `V` onto the
   lambda-return `R_t = r_{t+1} + gamma (1 - d_{t+1}) ((1 - lam) V_{t+1} + lam R_{t+1})`,
   `R_H = V_H`, with a few epochs of shuffled minibatches and no target network.

## 1. Components

| Class | File | Role |
|---|---|---|
| `SuperMimic`, `SuperMimicCore` | `rl_garden/algorithms/supermimic.py` | Rollout with noisy actions, observation-history ring, imagination, Eq. 1-3 updates, optimizers, sample reuse schedule, optional cosine learning-rate decay |
| `SuperMimicPolicy` | `rl_garden/algorithms/supermimic.py` | Goal encoder, observation-history decoder (the actor), state-value critic, input normalizers; holds the world model |
| `StateSpaceWorldModel` | `rl_garden/world_models/state_space.py` | Explicit-state dynamics, `model_loss` (Eq. 1), learned or analytic heads |
| `NominalErrorStateSpaceWorldModel` | `rl_garden/world_models/nominal_error_state_space.py` | Subclass that predicts raw increments and scores the state error in the codec's loss blocks with fixed nominal errors (the official loss) |
| `StateCodec`, `ResidualStateCodec` | `rl_garden/world_models/state_space.py` | What the dynamics MLP sees, what it predicts and how a prediction is applied |
| `LossBlockStateCodec` | `rl_garden/world_models/state_space.py` | A `StateCodec` that also names the loss space: `loss_blocks(s)` and `nominal_errors` |
| `AnalyticMDP` | `rl_garden/world_models/state_space.py` | Protocol for task-supplied, differentiable reward, observation, termination and goal |
| `GoalSource` | `rl_garden/world_models/state_space.py` | Protocol for batched, differentiable goals from state, context and time, independent of head mode |
| `StateSequenceReplayBuffer` | `rl_garden/buffers/state_sequence_replay_buffer.py` | Single-episode world-model windows, plus `sample_history` for imagination starts |
| `build_supermimic`, `SuperMimicArgs` | `rl_garden/training/online/entrypoints/supermimic.py` | CLI wiring and registration (`supermimic`) |

`SuperMimicPolicy` is a plain `nn.Module` outside the `BasePolicy` contract, like
`PWMPolicy`: the critic reads the world model's state features, the actor's imagined
observations come from the world model, and no observation encoder is trained. The world
model is saved under the checkpoint's `policy` namespace. Three optimizers (world model,
actor, critic) share one learning rate, constant by default. Setting `lr_final_ratio < 1`
enables cosine decay from `lr` to `lr * lr_final_ratio` over `lr_decay_updates` updates.
All dynamics, learned-head, actor and critic MLPs use Linear → LayerNorm → GELU hidden
layers and PyTorch's default initialization. The actor retains running observation and
goal normalization; the critic reads its state features directly, without input normalization.

## 2. Environment contract

The algorithm adds no env method. Everything is an observation key, declared with
`--obs.extra_state` (so `--obs.extra_state sim ctx goal time` makes the backend check that
the env emits `state_sim`, `state_ctx`, `state_goal` and `state_time`). Replay stores the
whole observation dict, so the privileged keys reach replay without any buffer change.
Keep `state_sim` to the minimal sufficient state: it grows replay memory as `(T, N, S)`.

| Key | Meaning | Required | Default when absent |
|---|---|---|---|
| `state` | Policy observation `o` (sensor noise already added; may contain the previous action) | yes | none |
| `state_sim` | Simulator state `s` the world model predicts | no | `state` (fully observed task) |
| `state_ctx` | Per-episode simulation context `xi` (constant within an episode) | no | zero width |
| `state_goal` | Goal `g_t` fed to the actor | no | zero width |
| `state_time` | Reference time index, an integer-valued float | for `--heads analytic`; time-dependent goal sources also need correct indices | zero |

Episode ends follow the repository's standard boundary contract (`final_observation`,
`terminated` versus `truncated`). Bootstrapping stops at terminations only. Like every
off-policy algorithm, replay keeps the pre-reset observation on every episode end, so the
last window target is the true final state.

`--heads learned` (default) trains the reward, termination and, if `state_sim` is present,
observation heads as small MLPs. With only `state`, the observation head is the identity.
No extra sensor noise is added to imagined observations in this mode.
Learned heads are evaluated on predicted states inside the k-step rollout, so their
losses also shape the dynamics (Eq. 1); their training inputs keep each step's recorded goal.

`--heads analytic` takes all three heads from the env's `AnalyticMDP`, found at
`env.unwrapped.analytic_mdp`, which also supplies imagination goals by default.
It raises `ValueError` at construction when the env has no `analytic_mdp`, lacks
`state_time` (a time-independent MDP still emits a zero column), or when the MDP's
`obs_dim`/`goal_dim` do not match the `state` and `state_goal` widths.

Goal generation is independent of head mode: pass `SuperMimic(goal_source=...)` or
expose `env.unwrapped.goal_source`. The explicit argument takes precedence, then the
env attribute; if neither is set, the world model falls back to its `AnalyticMDP`.
There is no CLI flag for this Python object. Each imagined step recomputes
`goal_source.goal(s_next, ctx, t_next)` without detaching the predicted state, so
state-relative goals carry gradients back through dynamics. A source's `goal_dim`
must match the `state_goal` width, including zero-width goals, or construction raises `ValueError`.

| Protocol | Goal signature | Contract |
|---|---|---|
| `GoalSource` | `goal(s, ctx, t) -> Tensor` | Batched, pure torch, differentiable; `(B, goal_dim)`, including `(B, 0)` |
| `AnalyticMDP` | `goal(s, ctx, t) -> Tensor` | Also satisfies `GoalSource`, alongside the task heads below |

```python
class StateCodec(Protocol):
    state_dim: int
    feature_dim: int
    increment_dim: int
    error_floor: Optional[Tensor]            # (increment_dim,) lower bound on the error scale
    def features(self, s) -> Tensor          # (..., S) -> (..., F), the MLP input
    def increment(self, s, s1) -> Tensor     # local increment from s to s1; error = increment(s1, s_hat)
    def integrate(self, s, d) -> Tensor      # s (+) d, differentiable in both arguments

class LossBlockStateCodec(StateCodec, Protocol):
    nominal_errors: Mapping[str, float]      # nominal error per block, keys == loss_blocks keys
    def loss_blocks(self, s) -> dict[str, Tensor]   # {name: (..., W_name)}, differentiable

class GoalSource(Protocol):                  # batched, pure torch, differentiable
    goal_dim: int
    def goal(self, s, ctx, t) -> Tensor      # (B, goal_dim), or (B, 0) for goal_dim == 0

class AnalyticMDP(GoalSource, Protocol):      # batched, pure torch, differentiable
    obs_dim: int
    goal_dim: int
    obs_noise: Optional[Tensor]              # (obs_dim,) half-width of uniform observation noise
    def observation(self, s, prev_action, ctx, t) -> Tensor
    def reward(self, s, action, s_next, prev_action, ctx, t_next) -> Tensor      # (B,)
    def termination_prob(self, s, t) -> Tensor                                   # (B,) in [0, 1]
    def goal(self, s, ctx, t) -> Tensor                                         # (B, goal_dim)
    def critic_features(self, s, ctx, t) -> Tensor                               # (B, K), K may be 0
    def valid_start(self, t, horizon) -> Tensor                                  # (B,) bool
```

Time convention: `reward(..., t_next)` receives `t_next = t + 1`, the index of
`s_next`; `termination_prob(s, t)` receives the post-step state and its index.
Official SuperMimic scores before advancing motion time, against `ref(t_next - 1)`
for reward and termination; subtract one when porting those heads. `valid_start`
may return all True, with motion time clamped at clip ends by the MDP as in the official code.

`ResidualStateCodec(state_dim)` is the default: features are the state, the increment is
`s1 - s`, integration adds. Custom codecs (for example a quaternion or SE(3) pose, whose
error should live in a local frame) are a Python-API argument, `SuperMimic(codec=...)`;
the CLI has no flag for it. With `heads="analytic"`, start rows are rejected unless
`mdp.valid_start(t, horizon)` holds (up to ten rounds of 2x oversampling).

Two concepts are separate: the **prediction target**, what the dynamics MLP outputs
(`increment`, applied by `integrate`), and the **loss space**, where the prediction error
is measured and how it is weighted. `StateSpaceWorldModel` ties them together: it
de-normalizes the MLP output with running increment statistics and scores
`increment(s_real, s_hat)` divided by the floored running increment std, summed over the
window. A `LossBlockStateCodec` additionally names the loss space: `loss_blocks(s)`
returns named channel groups `(..., W_g)` and `nominal_errors` one nominal error per group.
`NominalErrorStateSpaceWorldModel` uses them: the MLP predicts the raw increment (class
attribute `normalize_output = False`), and the state term is Bryson's rule

```text
diff_g = loss_blocks(s_hat)[g] - loss_blocks(s_real)[g]      # (k, B, W_g)
loss   = sum_g(mean(diff_g ** 2) * W_g / nominal_g ** 2) / sum_g(W_g)
```

with the mean over steps, batch and channels; channels outside every block do not enter
the loss, and `error_floor` is unused. Each block is also logged as `wm/state_block/<g>`
(`mean(diff_g ** 2) / nominal_g ** 2`). Construction raises `ValueError` unless the codec
has `loss_blocks` and finite positive `nominal_errors` whose keys equal the block names;
block widths come from one CPU probe `loss_blocks(zeros(1, state_dim))`. Select the class
with `SuperMimic(world_model_cls=NominalErrorStateSpaceWorldModel, codec=...)` or by
overriding the class attribute `SuperMimic.world_model_cls` in a subclass; the qualified
class name is checkpoint metadata, and there is no CLI flag. This is the official
GenCoord/MaxCoord scheme: velocity-level increments are predicted, poses are integrated, and
pose (and joint or body) blocks are scored with fixed nominal errors. No humanoid codec ships
in the tree.

## 3. Schedule and hyperparameters

The official schedule defines sample reuse via `replay_ratio` (finite and positive).
Each iteration collects one vectorized env step and adds
`r = num_envs * replay_ratio / (wm_batches * wm_batch_size * (wm_window + 1))`
to an update accumulator. Its integer part is trained and the fraction carries forward;
an iteration with no whole update returns empty metrics. Defaults give `r = 0.4` with
4096 envs. `wm_window=7` scores seven predictions from an eight-frame window.

Legacy scheduling requires `replay_ratio=None` and both explicit iteration parameters
(`sim_steps_per_iteration >= 1`, `updates_per_iteration >= 1`); it collects the specified
vector steps and runs the specified updates per iteration. Supplying either explicit
parameter in replay-ratio mode raises `ValueError`. rl-garden's own `utd` is per
collected transition and is derived internally.

Each update trains the world model on `wm_batches` windows (one Adam step each), then
the actor once, then the critic for `critic_epochs * critic_minibatches` steps.
World-model-only warmup defaults to zero; when enabled its duration is
`ceil(warmup_updates / r) * num_envs` transitions in replay-ratio mode and
`warmup_updates * sim_steps_per_iteration * num_envs` in legacy mode, counted from step 0.
Gradients are clipped by global norm and, for the actor, then element-wise; a step whose
gradient norm is not finite is skipped and counted (`actor/skipped`).

`max_step_reward` optionally caps the actor's terminal bootstrap and the completed
critic lambda-return targets at `max_step_reward / (1 - gamma)`. It leaves rewards and
the lambda-return recursion unchanged; `None` disables the cap. A finite reward bound
and `0 <= gamma < 1` are required when enabling it.

| Official implementation / option | Argument | Default |
|---|---|---|
| Imagination horizon `H` | `horizon` | 32 |
| World-model scored predictions `k` (8 frames) | `wm_window` | 7 |
| World-model batches per update | `wm_batches` | 10 |
| World-model batch size | `wm_batch_size` | 2048 |
| Actor (imagination) batch | `imagination_batch` | 4096 |
| Discount | `gamma` | 0.99 |
| Lambda | `lam` | 0.95 |
| Learning rate, final ratio, optional decay length | `lr`, `lr_final_ratio`, `lr_decay_updates` | 2e-4, 1.0 (constant), 250000 |
| Actor Adam betas | `actor_betas` | (0.9, 0.95) |
| Clip: world model, actor norm, actor value, critic | `wm_grad_norm`, `actor_grad_norm`, `actor_grad_value`, `critic_grad_norm` | 1.0, 1.0, 0.1, 1.0 |
| Critic epochs, minibatches | `critic_epochs`, `critic_minibatches` | 5, 4 |
| Exploration noise std | `action_noise_std` | 0.1 |
| World-model MLP | `wm_hidden_dims` | (2048, 2048, 2048) |
| Learned-head MLP | `head_hidden_dims` | (256, 256) |
| Goal encoder, latent, decoder, critic | `goal_encoder_dims`, `latent_dim`, `decoder_dims`, `critic_dims` | (2048, 1024, 512, 256), 32, (2048, 1024, 512, 256), (256, 128) |
| Sample reuse ratio | `replay_ratio` | 16.0 |
| Legacy vector steps, updates per iteration | `sim_steps_per_iteration`, `updates_per_iteration` | None, None |
| Replay size (1000 rows per env) | `buffer_size` | None → `1000 * num_envs` |
| World-model-only warmup | `warmup_updates` | 0 |
| Optional reward bound for value caps | `max_step_reward` | None |

Additional options: `obs_history_len` (1), `amp_dtype`
(`None` = fp32 with TF32 matmuls; `bf16` is optional).
`learning_starts` defaults to `(wm_window + 2) * num_envs` transitions, and exploration is
always the noisy policy (no random-action phase). `--total_timesteps` counts env
transitions; its default (1M) is only a placeholder, since the right budget is task-specific
(the paper's runs are on the order of 100M).

## 4. Commands

Learned heads on an env that emits the keys you declare (state-only works too). The
`gymnasium` backend and external backends can supply extra state keys; every other
in-tree backend rejects `--obs.extra_state` at environment construction with
"`<backend>` has no extra state sources":

```bash
rl-garden-train-online supermimic \
  --env_backend <backend> --env_id <env> --num_envs 4096 \
  --obs.extra_state sim ctx goal time
```

Analytic heads, with a task-supplied `AnalyticMDP` on the env:

```bash
rl-garden-train-online supermimic \
  --env_backend <backend> --env_id <env> --num_envs 4096 \
  --obs.extra_state sim ctx goal time --heads analytic \
  --obs_history_len 10 --buffer_size 4096000
```

`--print-config` validates the flags (it does not check that the backend emits the declared
keys), and `--dry-run` builds the env and the agent without
training. `--obs.extra_state` is the only observation flag that matters here: `--encoder.*`,
`--obs_groups.*`, `--critic_encoder.*` and `--encoder_sharing` have no effect because the
policy trains no observation encoder.

## 5. Known limitations

- With learned heads and time-varying goals but no `goal_source`, imagined reward,
  observation and termination heads read a goal frozen at the imagination start,
  while training sees stepwise advancing recorded goals. Construction emits a warning;
  provide a `GoalSource` to recompute goals during imagination.
- Single process, single GPU. There is no DDP path.
- The observation-history ring used while collecting (and while evaluating) is not
  checkpointed. After `--load_checkpoint` it restarts at the next env reset, with the first
  frame replicated.
- Fractional update credit is not checkpointed; resuming can differ by at most one
  update. The replay ratio and derived updates per vector step are checkpoint metadata.
- Optional world-model warmup is timed in collected transitions from step 0, including
  collection before `learning_starts`; it does not guarantee that many completed updates.
- Imagination start states come from the replay buffer only, that is, the most recent
  `buffer_size / num_envs` steps per env. There is no task-side reset sampler.
- A termination head learned on a task that never terminates collapses to a constant, and
  with very sparse terminations its BCE loss is biased towards "not terminated".

## 6. Defaults and differences from the paper and reference implementations

- Heads default to learned; the reference reproductions use analytic ones throughout.
  Analytic heads are a plug-in (`heads="analytic"`).
- The default world model scores the codec increment divided by its running std and
  de-normalizes the MLP output; the official implementation predicts unnormalized
  velocity-level increments and scores integrated pose blocks with fixed nominal errors
  (mean over the window rather than sum). That loss is available as
  `NominalErrorStateSpaceWorldModel` with a `LossBlockStateCodec`
  (`world_model_cls=...`); the default stays `StateSpaceWorldModel`.
- Defaults follow the official code: sample reuse ratio 16, eight-frame world-model
  windows, constant learning rates and zero world-model warmup. The paper's Table A.5
  lists cosine decay; it remains available by setting `lr_final_ratio < 1`.
- Updates per vector step scale with env count and sampled frames through `replay_ratio`;
  0.4 is the result for 4096 envs with the defaults. The explicit legacy schedule remains
  available with `replay_ratio=None`.
- Value caps are opt-in through `max_step_reward`, since the generic env contract does
  not supply a task's reward bound.
- Precision defaults to fp32 with TF32 matmuls, as in the paper; the reproductions use
  bf16. `amp_dtype="bf16"` selects it.
- The non-finite-gradient guard applies to all three optimizers, not only the actor.
- Imagination starts reuse the stored policy observation history (it already carries the
  env's sensor noise), so each frame is noised exactly once; only imagined frames get
  fresh noise, and only when an `AnalyticMDP` provides `obs_noise`.
- A learned observation head is trained on the noisy `state` observation, so it learns the
  zero-mean noise's mean.
- Episode ends of any kind keep their pre-reset observation in replay (the off-policy base
  class does this for every algorithm), so the world model's last window target is the true
  final state on terminations too. Bootstrapping still stops at terminations only.
- Imagined actions are clamped to the env's action bounds, so out-of-range actions receive
  zero gradient and the model sees the action distribution it was trained on.
- `heads="analytic"` requires `state_time` so imagination starts from the recorded
  reference index; otherwise every start would use zero and advance from there.
- Windows never cross episodes, so the reproductions' prefix mask on steps after `done` is
  not needed. There is no EMA copy of the policy.

## 7. References

- Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel,
  Koushil Sreenath. *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
  World Model.* CoRL 2026. arXiv:2610.28480, <https://arxiv.org/abs/2610.28480>; project page
  <https://supermimic.github.io>.
- Official implementation (the authority for defaults and MLP architecture):
  <https://github.com/HybridRobotics/supermimic>.
- Two independent reproductions informed the state representation
  and task interfaces: `supermimic_vibed` (Chong Zhang, ETH
  Zurich, MIT license, <https://github.com/zita-ch/supermimic_vibed>) and `Uni-SuperMimic`
  (Motphys, Apache-2.0, <https://github.com/Motphys/Uni-SuperMimic>).
