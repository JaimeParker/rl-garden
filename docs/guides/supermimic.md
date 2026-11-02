# SuperMimic: Backpropagation Through an Explicit-State World Model

SuperMimic (arXiv 2610.28480) trains a deterministic policy by backpropagating the
return through imagined rollouts of a differentiable world model. The model works on the
simulator's explicit state `s`, not on a learned latent. Each iteration collects a few
vectorized env steps with the noisy policy, then runs a few updates. An update has three
parts (the paper's Eq. 1-3):

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
| `SuperMimic`, `SuperMimicCore` | `rl_garden/algorithms/supermimic.py` | Rollout with noisy actions, observation-history ring, imagination, Eq. 1-3 updates, optimizers, cosine learning-rate schedule |
| `SuperMimicPolicy` | `rl_garden/algorithms/supermimic.py` | Goal encoder, observation-history decoder (the actor), state-value critic, input normalizers; holds the world model |
| `StateSpaceWorldModel` | `rl_garden/world_models/state_space.py` | Explicit-state dynamics, `model_loss` (Eq. 1), learned or analytic heads |
| `StateCodec`, `ResidualStateCodec` | `rl_garden/world_models/state_space.py` | What the dynamics MLP sees, what it predicts and how a prediction is applied |
| `AnalyticMDP` | `rl_garden/world_models/state_space.py` | Protocol for task-supplied, differentiable reward, observation, termination and goal |
| `StateSequenceReplayBuffer` | `rl_garden/buffers/state_sequence_replay_buffer.py` | Single-episode world-model windows, plus `sample_history` for imagination starts |
| `build_supermimic`, `SuperMimicArgs` | `rl_garden/training/online/entrypoints/supermimic.py` | CLI wiring and registration (`supermimic`) |

`SuperMimicPolicy` is a plain `nn.Module` outside the `BasePolicy` contract, like
`PWMPolicy`: the critic reads the world model's state features, the actor's imagined
observations come from the world model, and no observation encoder is trained. The world
model is saved under the checkpoint's `policy` namespace. Three optimizers (world model,
actor, critic) share one learning rate that decays by cosine from `lr` to
`lr * lr_final_ratio` over `lr_decay_updates` updates.

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
| `state_time` | Reference time index, an integer-valued float | only for `--heads analytic` | none |

Episode ends follow the repository's standard boundary contract (`final_observation`,
`terminated` versus `truncated`). Bootstrapping stops at terminations only. Every episode
end keeps its pre-reset observation, so the last window target is the true final state.

`--heads learned` (default) trains the reward, termination and, if `state_sim` is present,
observation heads as small MLPs. With only `state`, the observation head is the identity.
No extra sensor noise is added to imagined observations in this mode.

`--heads analytic` takes all three heads from the env's `AnalyticMDP`, found at
`env.unwrapped.analytic_mdp`, and also advances the goal in imagination with `mdp.goal(t)`.
It raises `ValueError` at construction when the env has no `analytic_mdp`, lacks
`state_time` (a time-independent MDP still emits a zero column), or when the MDP's
`obs_dim`/`goal_dim` do not match the `state` and `state_goal` widths.

```python
class StateCodec(Protocol):
    state_dim: int
    feature_dim: int
    increment_dim: int
    error_floor: Optional[Tensor]            # (increment_dim,) lower bound on the error scale
    def features(self, s) -> Tensor          # (..., S) -> (..., F), the MLP input
    def increment(self, s, s1) -> Tensor     # local increment from s to s1; error = increment(s1, s_hat)
    def integrate(self, s, d) -> Tensor      # s (+) d, differentiable in both arguments

class AnalyticMDP(Protocol):                 # batched, pure torch, differentiable
    obs_dim: int
    goal_dim: int
    obs_noise: Optional[Tensor]              # (obs_dim,) half-width of uniform observation noise
    def observation(self, s, prev_action, ctx, t) -> Tensor
    def reward(self, s, action, s_next, prev_action, ctx, t_next) -> Tensor      # (B,)
    def termination_prob(self, s, t) -> Tensor                                   # (B,) in [0, 1]
    def goal(self, t) -> Tensor                                                  # (B, goal_dim)
    def critic_features(self, s, ctx, t) -> Tensor                               # (B, K), K may be 0
    def valid_start(self, t, horizon) -> Tensor                                  # (B,) bool
```

`ResidualStateCodec(state_dim)` is the default: features are the state, the increment is
`s1 - s`, integration adds. Custom codecs (for example a quaternion or SE(3) pose, whose
error should live in a local frame) are a Python-API argument, `SuperMimic(codec=...)`;
the CLI has no flag for it. With `heads="analytic"`, start rows are rejected unless
`mdp.valid_start(t, horizon)` holds (up to ten rounds of 2x oversampling).

## 3. Schedule and hyperparameters

One iteration collects `sim_steps_per_iteration` vectorized env steps, then runs
`updates_per_iteration` updates. The paper's 0.4 updates per vectorized env step is
`sim_steps_per_iteration=5, updates_per_iteration=2` (rl-garden's own `utd` is per
transition, so it is not exposed). Each update trains the world model on `wm_batches`
windows (one Adam step each), then the actor once, then the critic for
`critic_epochs * critic_minibatches` steps. The first `warmup_updates` updates train only
the world model. Gradients are clipped by global norm and, for the actor, then element-wise;
a step whose gradient norm is not finite is skipped and counted (`actor/skipped`).

| Paper (Tables A.4 and A.5) | Argument | Default |
|---|---|---|
| Imagination horizon `H` | `horizon` | 32 |
| World-model window `k` | `wm_window` | 8 |
| World-model batches per update | `wm_batches` | 10 |
| World-model batch size | `wm_batch_size` | 2048 |
| Actor (imagination) batch | `imagination_batch` | 4096 |
| Discount | `gamma` | 0.99 |
| Lambda | `lam` | 0.95 |
| Learning rate, final ratio, decay length | `lr`, `lr_final_ratio`, `lr_decay_updates` | 2e-4, 0.03, 250000 |
| Actor Adam betas | `actor_betas` | (0.9, 0.95) |
| Clip: world model, actor norm, actor value, critic | `wm_grad_norm`, `actor_grad_norm`, `actor_grad_value`, `critic_grad_norm` | 1.0, 1.0, 0.1, 1.0 |
| Critic epochs, minibatches | `critic_epochs`, `critic_minibatches` | 5, 4 |
| Exploration noise std | `action_noise_std` | 0.1 |
| World-model MLP | `wm_hidden_dims` | (2048, 2048, 2048) |
| Learned-head MLP | `head_hidden_dims` | (256, 256) |
| Goal encoder, latent, decoder, critic | `goal_encoder_dims`, `latent_dim`, `decoder_dims`, `critic_dims` | (2048, 1024, 512, 256), 32, (2048, 1024, 512, 256), (256, 128) |
| Replay size | `buffer_size` | 4096000 |

Not in the paper's table: `obs_history_len` (1), `warmup_updates` (500, the reference
reproductions' choice), `amp_dtype` (`None` = fp32 with TF32 matmuls; `bf16` is optional).
`learning_starts` defaults to `(wm_window + 2) * num_envs` transitions, and exploration is
always the noisy policy (no random-action phase). `--total_timesteps` counts env
transitions; its default (1M) is only a placeholder, since the right budget is task-specific
(the paper's runs are on the order of 100M).

## 4. Commands

Learned heads on any env that emits the keys you declare (state-only works too):

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

`--print-config` validates the flags, and `--dry-run` builds the env and the agent without
training. `--obs.extra_state` is the only observation flag that matters here: `--encoder.*`,
`--obs_groups.*`, `--critic_encoder.*` and `--encoder_sharing` have no effect because the
policy trains no observation encoder.

## 5. Known limitations

- With learned heads, the goal stays at its start value during imagination: nothing
  learned can advance it. An `AnalyticMDP` provides `goal(t)` and removes the limitation.
- Single process, single GPU. There is no DDP path.
- The observation-history ring used while collecting (and while evaluating) is not
  checkpointed. After `--load_checkpoint` it restarts at the next env reset, with the first
  frame replicated.
- Imagination start states come from the replay buffer only, that is, the most recent
  `buffer_size / num_envs` steps per env. There is no task-side reset sampler.
- A termination head learned on a task that never terminates collapses to a constant, and
  with very sparse terminations its BCE loss is biased towards "not terminated".

## 6. Differences from the paper and the reference reproductions

- Heads default to learned; the reference reproductions use analytic ones throughout.
  Analytic heads are a plug-in (`heads="analytic"`).
- The world-model warmup length (500 updates) follows the reproductions; the paper does
  not state it. It is counted in env steps (`warmup_updates * sim_steps * num_envs`) from
  step 0, so it also covers the collection before `learning_starts`.
- The update-to-data ratio 0.4 is five env steps and two updates per iteration. The
  reproductions alternate between two and three steps per update, which averages 2.5.
- Precision defaults to fp32 with TF32 matmuls, as in the paper; the reproductions use
  bf16. `amp_dtype="bf16"` selects it.
- The non-finite-gradient guard applies to all three optimizers, not only the actor.
- Imagination starts reuse the stored policy observation history (it already carries the
  env's sensor noise), so each frame is noised exactly once; only imagined frames get
  fresh noise, and only when an `AnalyticMDP` provides `obs_noise`.
- A learned observation head is trained on the noisy `state` observation, so it learns the
  zero-mean noise's mean.
- Episode ends of any kind keep their pre-reset observation in replay (the base class would
  keep it for truncations only); the world model's last window target needs the true final
  state on terminations too. Bootstrapping is unchanged.
- Imagined actions are clamped to the env's action bounds, so out-of-range actions receive
  zero gradient and the model sees the action distribution it was trained on.
- `heads="analytic"` requires `state_time`; without it the time index would be zero
  throughout and the goal would never advance.
- Windows never cross episodes, so the reproductions' prefix mask on steps after `done` is
  not needed. There is no EMA copy of the policy.

## 7. References

- Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel,
  Koushil Sreenath. *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable
  World Model.* CoRL 2026. arXiv:2610.28480, <https://arxiv.org/abs/2610.28480>; project page
  <https://supermimic.github.io>.
- No official code was released at the time of writing. Two independent reproductions were
  consulted for the details the paper leaves open (state representation, increment
  integration, reset sampling, warm-up, normalizers): `supermimic_vibed` (Chong Zhang, ETH
  Zurich, MIT license, <https://github.com/zita-ch/supermimic_vibed>) and `Uni-SuperMimic`
  (Motphys, Apache-2.0, <https://github.com/Motphys/Uni-SuperMimic>).
