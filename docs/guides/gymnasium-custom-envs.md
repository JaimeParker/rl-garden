# Training on Your Own Gymnasium Envs

The `gymnasium` env backend trains rl-garden algorithms on any registered,
single-instance `gymnasium.Env` that lives in your own package. You do not
fork rl-garden and you do not write a backend. The backend calls
`gymnasium.make(--env_id)` once per parallel env, vectorizes the instances
itself, and adapts them to rl-garden's torch env contract.

Write a dedicated backend (see
[`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md))
only when the simulator is already batched (GPU sims), needs process-global
state such as one app per process, or needs named, cross-task CLI knobs.

## Quick start

Register the env in a module of your package:

```python
# my_pkg/envs.py
import gymnasium as gym

gym.register("MyReach-v0", entry_point="my_pkg.reach:MyReachEnv", max_episode_steps=200)
```

Train with the online entrypoint. `--gymnasium.import_modules` names the
module(s) whose import runs `gymnasium.register`:

```bash
python examples/train_online.py sac \
  --env_backend gymnasium \
  --env_id MyReach-v0 \
  --gymnasium.import_modules my_pkg.envs \
  --gymnasium.vectorization async
```

The YAML preset equivalent:

```yaml
env_backend: gymnasium
env_id: MyReach-v0
gymnasium:
  import_modules: [my_pkg.envs]
  vectorization: async
```

Programmatic use (`make_gymnasium_env` performs no registration of its own,
so import your module first or list it in `import_modules`):

```python
import my_pkg.envs  # noqa: F401
from rl_garden.algorithms import SAC
from rl_garden.envs import GymnasiumEnvConfig, make_gymnasium_env

env = make_gymnasium_env(GymnasiumEnvConfig(env_id="MyReach-v0", num_envs=8, seed=0))
SAC(env=env, buffer_device="cuda").learn(total_timesteps=100_000)
```

## Settings (`--gymnasium.*`)

| Field | Default | Meaning |
|---|---|---|
| `import_modules` | `()` | Modules imported in every env worker before `gym.make` |
| `env_kwargs_json` | `"{}"` | JSON object forwarded verbatim to `gym.make(env_id, **kwargs)` |
| `vectorization` | `"sync"` | `sync` (`SyncVectorEnv`, one process) or `async` (`AsyncVectorEnv`, one process per env) |
| `device` | `"cpu"` | Device of the observation/reward tensors the env returns |

Reward scaling uses the shared `--reward_scale`/`--reward_bias` flags, and
`--num_envs`/`--num_eval_envs`/`--seed` work as for every backend. A
`reset()` without a seed replays the configured seed, so each evaluation
round uses the same episodes.

## Why `import_modules`, and not `module:EnvId`

Gymnasium also accepts `--env_id my_pkg.envs:MyReach-v0`, which imports
`my_pkg.envs` inside `gym.make`. The gymnasium backend accepts that form too,
but prefer `import_modules`:

- The env id becomes part of the run name, log group, log directory and
  checkpoint paths. A `:` in those paths breaks `rsync host:path` and
  Windows paths.
- `import_modules` keeps `--env_id` equal to the registered id and can
  import several modules.

Each listed module is imported *inside* the env thunk, in whichever process
calls `gym.make`. This matters for `async`: under a `spawn` or `forkserver`
start method (the default on macOS, and from Python 3.14 on Linux), worker
processes do not inherit the parent's imports. Importing your module in the
training script is therefore not enough. `--print-config` never imports
these modules; they are imported only when an env is built.

## Observation keys

Algorithms consume Dict observations whose keys use one strict vocabulary
(`rl_garden.observations.schema`). Your env must emit these keys itself:

| Key | Space |
|---|---|
| `state`, `state_<name>` | 1-D float `Box` |
| `rgb_<cam>` | `uint8` `Box`, shape `(H, W, 3)` |
| `depth_<cam>` | `float32` `Box`, shape `(H, W, 1)` |

One shortcut applies: a bare 1-D float `Box` observation is wrapped into
`Dict({"state": Box})`. Any other bare `Box`, such as an image, is rejected
with an `ObservationContractError` asking for a Dict observation.

After construction, the registry checks that the env's keys equal exactly
the keys `--obs.*` asks for: image keys from `--obs.rgb`/`--obs.depth`, then
`state` unless `--obs.state false`, then `state_<name>` from
`--obs.extra_state`. For an env that emits `rgb_front` and `state`:

```bash
python examples/train_online.py drqv2 \
  --env_backend gymnasium --env_id MyPixelReach-v0 \
  --gymnasium.import_modules my_pkg.envs \
  --obs.rgb front --gymnasium.vectorization async
```

The backend never renames keys and never resizes images. `--obs.image_size`
must stay unset, because the env's own resolution is used.
`--obs.frame_stack N` stacks every requested image key to
`(N_envs, N, H, W, C)` and requires `--obs.rgb` or `--obs.depth`.

## Episode boundaries and statistics

The backend vectorizes with `autoreset_mode=SAME_STEP`. On the step an
episode ends, `infos["final_observation"]` holds the true last observation
and `infos["final_info"]` holds the last step's info, so bootstrapping and
logging stay exact. Training and evaluation log `final_info["episode"]`,
which `gymnasium.wrappers.RecordEpisodeStatistics` produces. The backend
adds that wrapper unless your env's wrapper chain already has one; a second
copy would fail on its own info key. If your entry point applies it
itself, it must also apply the `TimeLimit` inside it:
`gymnasium.register(..., max_episode_steps=N)` puts `TimeLimit` *outside*
your entry point's wrappers, so your statistics wrapper would never see the
truncation and `final_info` would lack `"episode"`. The simplest option is
to leave the wrapper to rl-garden.

## Sync vs async

- `sync`: all envs step in the training process. This is fastest for cheap,
  state-only envs.
- `async`: one OS process per env. Use it when a step is expensive or the
  env renders: each worker owns its own renderer or GL context.

Both paths return CPU torch tensors (`--gymnasium.device cpu`). The
algorithm's `--buffer_device` and update device stay independent.

## Offline and off2on

Offline pretraining takes its observation and action spaces from the
dataset, and builds an env only for optional live evaluation (through
`make_evaluation_env`). Off2on builds train and eval envs like the online
path. Both accept `--env_backend gymnasium` with the same `--gymnasium.*`
flags. The dataset's observation keys must match the keys your env emits.

## Backend-specific named knobs

`EnvBackendArgs` is a closed schema. A run cannot add fields to
`--gymnasium.*`, and per-task values belong in `env_kwargs_json`. If an
external env family needs its own named, cross-task CLI knobs, register a
backend from your package through the `rlgarden.env_backends` entry-point
group, and add its config by subclassing an algorithm's Args class in your
own launcher:

```python
from dataclasses import dataclass, field

from my_pkg.backend import MyBackendConfig  # backend registered with config_field="my_backend"
from rl_garden.training.online import registry
from rl_garden.training.online.sac import SACArgs, run_sac


@dataclass
class MySACArgs(SACArgs):
    my_backend: MyBackendConfig = field(default_factory=MyBackendConfig)


registry.register("my_sac", MySACArgs, run_sac)
registry.run_cli()  # python my_launcher.py my_sac --env_backend my_backend --my_backend.<field> ...
```

The field name must equal the backend's `config_field`, because the registry
reads the config with `getattr(args, config_field)`.
