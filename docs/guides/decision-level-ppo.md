# Decision-Level (Semi-Markov) PPO

Some environments are stepped and queried on every step, but the sampled action
takes effect on only some of them: a macro-action that runs for a variable
number of steps, an event-triggered controller, or a multi-rate controller whose
action components update at different times. A plain PPO treats every step as a
decision, so the steps whose action had no effect still enter the policy loss
and dilute the advantage statistics.

`--decision-level` makes PPO learn from the decision steps only. The env says
which steps those are through an optional info key, and PPO computes returns,
advantages and the policy loss accordingly. Everything else about the run
(rollout length, `--total_timesteps`, `--gamma`, logged episode statistics) is
counted in env steps, as before.

Supported by `ppo`, `recurrent_ppo` and `transformer_ppo`.

## 1. When to use

Use it when the policy is queried every step but its action is consumed only on
some steps, and the pattern depends on the env (per env, per episode, per
state).

It is not `ActionRepeatWrapper`. The two answer different cases:

| | `ActionRepeatWrapper` | decision-level PPO |
|---|---|---|
| Hold length | fixed `k`, same for all envs | variable, per env |
| Policy queried | once per `k` env steps | every env step |
| `--gamma` discounts | per decision (`k` env steps) | per env step |
| First step of an episode | always a decision | always a decision (required) |
| Rollout / log units | decisions | env steps |

If every action is held for the same fixed `k` steps, `ActionRepeatWrapper` is
simpler and cheaper. The two cannot be combined (see
[Flags](#4-flags-and-incompatible-combinations)).

## 2. Env contract

The authoritative statement is rules 9 and 10 of the episode boundary contract
in [`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md);
the consumer-side validation is `rl_garden/algorithms/_decision_mask.py`. In
short, `step()` (never `reset()`) returns one of two optional keys in `infos`:

| key | dtype / shape | row `i` (column `j`) is true iff |
|---|---|---|
| `infos["decision"]` | `torch.bool`, `(N,)` | the action passed to this `step()` took effect in env `i` |
| `infos["decision_components"]` | `torch.bool`, `(N,) + single_action_space.shape` | action component `j` took effect in env `i` |

Requirements:

- **Device.** On the env device; the consumer moves it to the algorithm device.
- **Every step or never.** A run reports its key on every step or on none.
  Absent everywhere means every step is a decision (the default behavior). A
  key that appears on only some steps, a key that switches between
  `decision` and `decision_components`, or both keys on one step raises
  `ValueError`, as does a wrong dtype or shape.
- **One key per run.** With `decision_components`, a step is a decision iff any
  component took effect.
- **SAME_STEP boundaries.** On the step that produced `terminated`/`truncated`
  the flag belongs to the episode that ended (the action given to that step),
  not to the episode that follows the autoreset.
- **The first step of every episode is a decision.** This is a documented
  requirement covered by tests; consumers do not check it per step. Without it
  the first observation of an episode would never be trained on.
- **`decision_components` validity.** The mask `M` may depend on the state and
  on the values of the components it marks as effective, never on the values of
  the components it marks as ineffective. Then, under a diagonal Gaussian
  policy, the product of the effective components' densities is exactly the
  likelihood of what the env executed. If `M` peeked at an ineffective
  component's value, that likelihood would be biased.

Backend support:

| backend | behavior |
|---|---|
| Isaac Lab adapter and the pass-through wrappers (e.g. `ImageFrameStackWrapper`) | forward both keys unchanged |
| gymnasium `TorchVectorEnvAdapter` | converts gymnasium's batched numpy value and its `_decision`/`_decision_components` presence mask into one dense `(N, ...)` `torch.bool` tensor; an ended row reads the flag from `final_info`; raises `ValueError` if any row did not report |
| `ActionRepeatWrapper`, `ActionChunkWrapper`, `SkillActionWrapper` (via `ChunkAccumulator`) | raise `ValueError`: one outer action spans several inner steps, so it has no defined flag |
| ManiSkill, mujoco_warp, RoboTwin | do not emit the keys (their `infos` are rebuilt on done steps, built by the backend, or list-valued) |

Backend how-tos: [Isaac Lab](isaaclab-custom-tasks.md#decision-steps),
[gymnasium envs](gymnasium-custom-envs.md#decision-steps).

If the env emits a key but `--decision-level` is off, PPO warns once and treats
every step as a decision. With `--decision-level` on and an env that never
reports a key, training is the plain path.

## 3. What changes in training

Notation: `d_t` is true if step `t` is a decision. A hold is the run of steps
`t+1 ... t+n-1` that follow decision `t` before the next decision at `t+n`.

### Returns and advantages (SMDP GAE)

Rewards are discounted by `gamma` on every env step, and `lambda` is applied
once per decision. Per step, with `delta_t = r_t + gamma V_{t+1} - V_t`:

```
A_t = delta_t + gamma * lam_t * A_{t+1},    lam_t = lambda if d_{t+1} else 1
```

Inside a hold `lam_t = 1`, so the `V` terms of the hold's steps cancel and the
advantage of a decision step is

```
A_t = sum_{k<n} gamma^k r_{t+k} + gamma^n (V_{t+n} + lambda * A_{t+n}) - V_t
```

- An episode that ends inside a hold is cut by the done flag as usual
  (truncations still bootstrap from `final_observation`, see
  `--bootstrap-at-done`).
- A hold that runs past the end of the rollout window bootstraps from
  `last_values`, the value of the observation after the last stored step. The
  next window starts inside that hold; its first rows are not decisions.
- Implemented by `compute_gae(..., step_lambda=...)` in
  `rl_garden/buffers/gae.py` and wired in `RolloutBuffer`.

### Losses

Only decision rows enter the policy side of the update: the clipped surrogate,
the entropy bonus, `approx_kl`/`old_approx_kl`/`clipfrac`, the adaptive-KL
learning-rate rule, and the advantage normalization (mean and unbiased std over
the decision rows of the minibatch). Each is a mean over the minibatch's
decision count, not over its size. Non-decision rows are replaced by finite
values before any arithmetic (`torch.where`), so they contribute exactly zero
to the loss and gradients even if non-finite.

- **Zero-decision minibatch.** All masked means are 0 (no division by zero),
  advantage normalization is a no-op for fewer than two decision rows,
  `approx_kl` is 0 so `--target-kl` never stops on it, and no NaN reaches the
  parameters. The value loss of such a minibatch still trains the critic when
  `--per-step-critic` is on.
- **DDP.** Each rank normalizes by its own decision count; gradients and the
  adaptive-KL statistic are all-reduced as in the plain path.
- **`decision_components`.** The policy returns per-component log-probs and
  entropies. The log-ratio, entropy and adaptive-KL terms of a step sum the
  components that took effect, so a component that never takes effect gets
  exactly zero policy gradient (and stays at its initial parameters under
  training). The behaviour log-prob stored for a step is the same sum, computed
  once per window. A decision step is any step on which at least one component
  took effect, and GAE uses those decisions.
- **Recurrent / transformer PPO.** Minibatches still split along the env axis
  and BPTT runs across the whole window, so the loss of a decision reaches the
  earlier non-decision steps of its hold through the sequence model. With the
  default `encoder_sharing=shared_critic_grad` the actor's features are
  detached and the policy loss does not reach the encoder; only
  `encoder_sharing=shared` sends it there.

### Critic (`--per-step-critic`, default on)

On: the critic is trained on every step. The target of a non-decision step is
the remaining-hold lambda-return: the discounted rewards up to the next
decision plus `gamma^m (V_next + lambda * A_next)`, computed by the same
recurrence (for a step inside a hold, `R_t = r_t + gamma * R_{t+1}`). Value
clipping, when on, uses the same row set.

Off: the critic is trained on decision rows only.

Why on by default: the rollout bootstraps from the value of non-decision states.
A window usually ends mid-hold (`last_values`), and a truncation can land
mid-hold (`final_observation`); a critic trained only at decisions is never fit
where those bootstraps read it. With an all-true mask both settings give the
same value loss. `train/explained_variance` is computed over the rows the
critic is trained on.

## 4. Flags and incompatible combinations

| flag | default | meaning |
|---|---|---|
| `--decision-level` | off | read the env's decision keys; SMDP GAE and decision-masked policy loss |
| `--per-step-critic` / `--no-per-step-critic` | on | train the critic on every step / on decision steps only (no effect when the window has no decision mask) |
| `--log-std-init-per-dim a b ...` | none | one initial log-std per action dimension; overrides `--log-std-init` |
| `--action-mean-bias-init a b ...` | none | initial bias of the actor's mean head, one per action dimension |
| `--action-mean-head-gain g` | none | orthogonal mean-head weights with gain `g` and zero bias |

The last three are independent of decision mode and apply to any PPO-family
run (`DiagGaussianActor`). Together they set the initial exploration noise per
action component and an initial mean that the bias actually controls: the mean
head is a plain `nn.Linear` that `--kernel-init` does not touch (it only
initializes the trunk), so a small-gain orthogonal head makes the initial mean
close to the bias instead of a random function of the features. The bias is
applied after the gain. `--action-mean-head-gain` consumes extra RNG, so modules
built after the actor initialize differently from a run without it. A list whose
length is not the action dimension raises `ValueError`.

```bash
python examples/train_online.py ppo --env_backend isaaclab --env_id <task> \
  --decision-level \
  --log-std-init-per-dim -1.0 -0.5 -0.5 \
  --action-mean-bias-init 0.0 0.5 0.5 --action-mean-head-gain 0.01
```

`decision_level` and `per_step_critic` are recorded in the checkpoint metadata.

Incompatible combinations:

- `gail`: `ValueError` at construction.
- `--finite-horizon-gae`: `ValueError` at construction (and in the buffer).
- `ActionRepeatWrapper`, `ActionChunkWrapper`, `SkillActionWrapper`
  (including `--isaaclab.action_repeat > 1`): the wrapper raises on the first
  step that carries a decision key.

## 5. Metrics

Logged only with `--decision-level`, and only for a window whose steps carried
the key:

| metric | meaning |
|---|---|
| `train/decisions` | decision steps in the window (over all envs) |
| `train/decision_fraction` | decisions / env steps in the window |
| `train/env_steps_per_decision` | env steps / decisions (window average) |
| `losses/decisions_per_minibatch_min`, `..._mean` | smallest / mean decision count over the update's minibatches; a minimum of 0 flags a minibatch with no policy signal |
| `train/component_usage/<j>` | fraction of env steps on which component `j` took effect (`decision_components` only) |
| `train/components_per_decision` | mean number of components that took effect per decision step (`decision_components` only) |
| `policy/log_std/<j>` | log-std of action dimension `j` after the update |

Unchanged: `_global_step`, `--total_timesteps`, `--num_steps`, `--log_freq`,
`train/return`, `train/episode_len` and the other episode statistics are all
per env step, and the existing `losses/*` entries (policy, value, entropy, KL,
clipfrac) are now means over decision rows (policy side) as described above.

## 6. Bit-identity guarantee

With `--decision-level` off, or on but with an env that reports no decision key,
training is bit-identical to a run without the feature. The init options are
bit-identical to a plain `nn.Linear` mean head and a uniform `log_std_init` when
left unset. Covered by:

- `tests/test_decision_level_gae.py`: `compute_gae` without `step_lambda`
  (bitwise and hard-coded values), `decision_level` with no env key,
  feature-off with an emitting env (warns once, ignored).
- `tests/test_decision_level_losses.py`: training without a decision mask,
  all-true mask versus the unmasked update, zero-decision minibatches, a hold
  spanning two windows, BPTT reach, CLI wiring, metrics.
- `tests/test_decision_components.py`: per-component rollout sums bitwise,
  training without a component mask, zero gradient for never-effective
  components, component metrics.
- `tests/test_actor_init_options.py`: defaults equal a plain mean head and a
  scalar log-std, checkpoint metadata round-trip, CLI wiring.
