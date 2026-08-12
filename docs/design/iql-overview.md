# IQL Implementation Summary

## Overview

`IQLCore` (`rl_garden/algorithms/iql.py`) implements Implicit Q-Learning: expectile
value-function regression plus advantage-weighted regression (AWR) for the actor. Two concrete
classes share this core:

- **`IQL`** (`OfflineRLAlgorithm`) — pure offline pretraining.
- **`_IQLRolloutTrainingShell`** (`OffPolicyAlgorithm`) — rollout-capable, backs **`Off2OnIQL`**
  for offline pretrain + online fine-tune, matching `Off2OnCalQL`'s preset (no warmup, offline
  data retained and mixed throughout online fine-tuning).

Compared against `3rd_party/implicit_q_learning` (official JAX/Flax repo), `3rd_party/CORL`
(`algorithms/offline/iql.py`), and `3rd_party/wsrl` (`wsrl/agents/iql.py`) directly against the
code, not from memory — see verification notes inline below.

## Where rl-garden already matches all three references (no flags needed)

- **Expectile value loss**: `L_τ(u) = |τ − 1[u<0]|·u²` on `u = min_i Q_target_i(s,a) − V(s)`,
  using dataset actions and the **target** critic. Identical formula and target-network usage in
  all four implementations.
- **Critic TD target**: `r + γ·(1−done)·V(s')`, bootstrapped off the **live** (non-target) value
  network — none of the four keep a separate target-V; only the critic has a Polyak-averaged copy.
- **AWR actor loss**: `exp_adv = clip(exp(adv·β), max=100)`, `actor_loss = −mean(exp_adv·log_prob)`,
  `adv` reused from the value loss's own `target_Q − V` computation, always detached.
- **Q used for `adv` comes from the target critic**, not the live one, in all four.
- **No observation normalization** — matches JAX and WSRL exactly. (CORL is the outlier here,
  with dataset mean/std normalization on by default; rl-garden intentionally does not match CORL
  on this axis.)
- Network shape `[256, 256]` MLP, ReLU, matches everywhere by default.

## Aligning with official IQL JAX / CORL / WSRL

JAX and CORL agree closely with each other (same expectile/temperature per environment, same
tanh-bounded-mean actor shape, same actor-only cosine LR default). **WSRL diverges from both** on
three axes — default `std_parameterization`, default actor LR schedule, and (see Known gap below)
the exact actor-mean shape — so this splits into two recipes rather than one shared flag block,
the same way `docs/design/wsrl-overview.md`'s Cal-QL section does for CQL/Cal-QL.

### Parameter reference

| Axis | Official IQL JAX | CORL IQL | WSRL IQL | rl-garden flag |
|---|---|---|---|---|
| Expectile / temperature (AntMaze) | 0.9 / 10.0 | 0.9 / 10.0 (`configs/offline/iql/antmaze/*.yaml`) | 0.9 / 10.0 | `--expectile 0.9 --temperature 10.0` |
| Expectile / temperature (locomotion/MuJoCo) | 0.7 / 3.0 | 0.7 / 3.0 (`TrainConfig` default) | 0.7 / 3.0 | `--expectile 0.7 --temperature 3.0` (rl-garden's own default) |
| AWR exp-advantage clip | 100.0 | 100.0 | 100.0 (`adv_clip_max`) | `--adv_clip_max 100.0` (rl-garden default) |
| Actor distribution shape | tanh-bounded mean, **no** Jacobian correction (`means=tanh(means)`, `tanh_squash_distribution=False`) | same (`GaussianPolicy`, mean has `nn.Tanh` output activation) | **mean is not tanh-bounded at all** when `tanh_squash_distribution=False` (`wsrl/networks/actor_critic_nets.py::Policy`) — no Jacobian correction either | `--actor_distribution unsquashed` reproduces the **JAX/CORL** shape (tanh-bounded mean via `tanh_mean=True`); **no flag reproduces WSRL's own shape** — see Known gap |
| Std parameterization | state-independent (`state_dependent_std=False`, fixed `log_stds` param) | state-independent (`nn.Parameter(torch.zeros(act_dim))`) | state-dependent (`std_parameterization="exp"`, default) | `--std_parameterization uniform` for JAX/CORL; rl-garden's own default (`exp`) already matches WSRL |
| Actor LR schedule | actor-only cosine decay to 0 over `max_steps`, **on by default** (`opt_decay_schedule="cosine"`) | same (`CosineAnnealingLR(actor_optimizer, max_steps)`, on by default) | off by default (`cosine_decay_steps=None`) | `--actor_lr_schedule warmup_cosine --actor_lr_decay_steps <total_grad_steps>` for JAX/CORL; rl-garden's own default (constant) already matches WSRL |
| Critic ensemble | double-Q, `n_critics=2` | double-Q (`TwinQ`, hardcoded to 2, not configurable) | double-Q, `critic_ensemble_size=2` | `--n_critics 2` and omit `--critic_subsample_size` — rl-garden's **CLI default is `n_critics=10, critic_subsample_size=2`** (a REDQ-style knob, intentionally left as-is per an earlier design review; override for reference parity) |
| AntMaze reward transform | `rewards -= 1.0` | `rewards -= 1.0` (same `modify_reward` shape, confirmed via `configs/offline/iql/antmaze/umaze_v2.yaml`: `normalize_reward: true`) | `reward*scale+bias`, antmaze launch script uses `scale=10.0, bias=-5.0` | `--reward_scale 1.0 --reward_bias -1.0` for JAX/CORL; `--reward_scale 10.0 --reward_bias -5.0` for WSRL. rl-garden's **default is a no-op** (`1.0`/`0.0`) — must be set explicitly either way |
| Discount / target update | gamma=0.99, tau=0.005 | gamma=0.99, tau=0.005 | gamma=0.99, tau=0.005 | rl-garden's own defaults already match |
| Actor/critic/value LR | 3e-4 each | 3e-4 each | 3e-4 each | rl-garden's own defaults already match |

### JAX/CORL recipe — AntMaze

```bash
python examples/pretrain_offline.py iql \
    --env_backend d4rl_legacy --dataset_backend d4rl_legacy \
    --expectile 0.9 --temperature 10.0 \
    --actor_distribution unsquashed --std_parameterization uniform \
    --actor_lr_schedule warmup_cosine --actor_lr_decay_steps 1000000 \
    --n_critics 2 \
    --reward_scale 1.0 --reward_bias -1.0 \
    --num_offline_steps 1000000 \
    --num_eval_episodes 100 --eval_episode_horizon 1000
```

`--n_critics 2` implicitly drops critic subsampling (`critic_subsample_size` only applies when
`< n_critics`). Needs the legacy D4RL AntMaze backend, not the Minari-recovered one — install the
`d4rl-legacy` extra first, same requirement as the Cal-QL recipes in `wsrl-overview.md`.

### WSRL recipe — AntMaze

Mostly rl-garden's own defaults already: `std_parameterization="exp"` and `lr_schedule="constant"`
need no override to match WSRL. The two real deltas are the reward transform and the critic
ensemble default:

```bash
python examples/pretrain_offline.py iql \
    --env_backend d4rl_legacy --dataset_backend d4rl_legacy \
    --expectile 0.9 --temperature 10.0 \
    --n_critics 2 \
    --reward_scale 10.0 --reward_bias -5.0 \
    --num_offline_steps 1000000 \
    --num_eval_episodes 100 --eval_episode_horizon 1000
```

Leave `--actor_distribution` at its default (`squashed`) or set it to `unsquashed` — neither
exactly reproduces WSRL's own actor shape; see Known gap below.

### Known gaps

- **Actor-mean shape: WSRL's own "unsquashed" variant is not reachable via a flag.** rl-garden's
  `actor_distribution="unsquashed"` always builds `UnsquashedGaussianActor(tanh_mean=True)`,
  matching JAX/CORL's tanh-bounded-mean shape. WSRL's `Policy` with `tanh_squash_distribution=False`
  applies **no** tanh to the mean at all (verified directly in
  `3rd_party/wsrl/wsrl/networks/actor_critic_nets.py`) — a third, distinct shape that
  `UnsquashedGaussianActor` itself already supports (`tanh_mean=False`, its original AWAC-parity
  default) but `IQLPolicy` does not currently expose as a choice. Non-blocking (both
  `actor_distribution` settings train and evaluate correctly), but neither setting is a bit-exact
  match for WSRL's own actor math. Ask if you need this closed — it would be a small additive
  change (thread `tanh_mean` through `IQLPolicy`/`IQLCore` as its own flag, or fold it into
  `actor_distribution` as a third value).
- **`evaluate_action_log_prob` (unsquashed mode) has no `action_scale`/`action_bias` term.** It
  scores actions in raw env coordinates, matching CORL/JAX's own math only when the action space is
  exactly `[-1, 1]` (true for the D4RL/AntMaze/MuJoCo spaces these recipes target). Not a bug —
  this mirrors the reference repos' own implicit assumption — but worth knowing before pointing it
  at a differently-scaled action space.
- No blocking gaps: unlike the off2on Cal-QL parity recipes (`wsrl-overview.md`'s Gap 1), an
  IQL offline-pretrained checkpoint built with either recipe above loads cleanly through
  `examples/train_off2on.py iql --load_checkpoint ...` — `Off2OnIQL` needs no algorithm-specific
  override at the offline→online switch (confirmed in `off2on_iql.py`'s own module docstring), and
  every flag in the tables above (`actor_distribution`, `std_parameterization`,
  `actor_lr_schedule`/`actor_lr_decay_steps`, `n_critics`) is wired through the off2on entrypoint
  identically to the offline one.

## Verification

```bash
pytest tests/test_iql.py tests/test_off2on_iql.py tests/test_off2on_iql_cli.py tests/test_cli_args.py -k iql
```

`tests/test_iql.py` includes direct checks that `actor_distribution="unsquashed"` builds
`UnsquashedGaussianActor(tanh_mean=True)`, that `actor_lr_schedule` genuinely decouples the actor
optimizer's LR from the critic/value optimizer's, and that a checkpoint saved without an actor
schedule loads cleanly into an agent configured with one.
