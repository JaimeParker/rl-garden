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
| `obs_key_map` | `()` | `src=dst` entries renaming the env's Dict observation keys to rl-garden keys (see [Renaming observation keys](#renaming-observation-keys)) |
| `vector_env_factory` | `""` | `pkg.module:function` that builds the whole vector env instead of `gym.make` (see [Bringing your own vector env](#bringing-your-own-vector-env)) |
| `vector_env_returns_torch` | `false` | The factory's vector env already returns torch tensors on `device`; requires `vector_env_factory` |

Reward scaling uses the shared `--reward_scale`/`--reward_bias` flags, and
`--num_envs`/`--num_eval_envs`/`--seed` work as for every backend; logged
episode `return` stays the sum of raw env rewards, before scale and bias. A
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

The backend renames keys only when `obs_key_map` asks it to (next section),
and it never resizes images. `--obs.image_size` must stay unset, because the
env's own resolution is used.
`--obs.frame_stack N` stacks every requested image key to
`(N_envs, N, H, W, C)` and requires `--obs.rgb` or `--obs.depth`.

## Renaming observation keys

If your env emits a Dict observation with its own key names, map them to
rl-garden keys with `obs_key_map` instead of editing the env. Each entry is
`src=dst`. Keys not in the map pass through unchanged. For an env that emits
`{"pixels": ..., "proprio": ...}`:

```bash
python examples/train_online.py drqv2 \
  --env_backend gymnasium --env_id MyPixelReach-v0 \
  --gymnasium.import_modules my_pkg.envs \
  --gymnasium.obs_key_map pixels=rgb_front proprio=state \
  --obs.rgb front
```

```yaml
env_backend: gymnasium
env_id: MyPixelReach-v0
obs:
  rgb: [front]
gymnasium:
  import_modules: [my_pkg.envs]
  obs_key_map: [pixels=rgb_front, proprio=state]
```

The rename applies once to the vectorized env, after the numpy-to-torch
conversion. It covers the observation spaces, every `reset`/`step`
observation and `infos["final_observation"]`, on both the `gym.make` path
and the `vector_env_factory` path. Frame stacking runs after the rename, so
`--obs.frame_stack` works on the renamed `rgb_front`. A bare 1-D float `Box`
is still auto-wrapped as `state` before the rename.

Checks:

- When the backend resolves its config (before any env is built, without
  importing anything), every entry is checked. Each entry needs exactly one
  `=` with non-empty sides. Sources and
  destinations must be unique. Every destination must be one of the keys
  `--obs.*` asks for.
- At construction, every source must be a key of the env's Dict observation
  space. A destination must not collide with a key the env already emits and
  that is not itself renamed. A non-Dict observation space is rejected.

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
algorithm's `--buffer_device` and update device stay independent. Only a
torch-native `vector_env_factory` (next section) returns tensors on another
device.

## Bringing your own vector env

When your env is already batched, or you want to control vectorization
yourself, point `vector_env_factory` at a function that builds the whole
vector env. The backend then calls it instead of `gym.make` +
`SyncVectorEnv`/`AsyncVectorEnv`:

```python
# my_pkg/vec.py
import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics


def make_vec(num_envs: int, seed: int, device: str, **env_kwargs) -> gym.vector.VectorEnv:
    return SyncVectorEnv(
        [lambda: RecordEpisodeStatistics(gym.make("MyReach-v0", **env_kwargs))] * num_envs,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
```

```bash
python examples/train_online.py sac \
  --env_backend gymnasium --env_id MyReach-v0 \
  --gymnasium.import_modules my_pkg.envs \
  --gymnasium.vector_env_factory my_pkg.vec:make_vec
```

```yaml
env_backend: gymnasium
env_id: MyReach-v0
gymnasium:
  import_modules: [my_pkg.envs]
  vector_env_factory: my_pkg.vec:make_vec
  vector_env_returns_torch: false
```

The factory contract is
`fn(num_envs: int, seed: int, device: str, **env_kwargs) -> gymnasium.vector.VectorEnv`,
where `env_kwargs` comes from `env_kwargs_json`. The backend imports
`import_modules` first, in the training process, then the factory's module.
Neither is imported before an env is built. `--env_id` is used only for naming
(run name, log and checkpoint paths); the factory chooses what to build.
`vectorization` must stay `sync`, because the factory does its own
vectorization.

The returned vector env must meet three requirements, checked at
construction:

- `metadata["autoreset_mode"]` is `AutoresetMode.SAME_STEP`. Any other mode
  is rejected, because bootstrapping relies on `final_observation` arriving
  on the step the episode ends.
- Its observation space is a `Dict`. A bare `Box` is rejected; it is not
  auto-wrapped as `state` on this path. Use `obs_key_map` to rename
  non-contract keys.
- It reports `final_info["episode"]` (`r`, `l`) itself. The backend does not
  add `RecordEpisodeStatistics` on this path. For a numpy vector env, wrap
  each single env in `RecordEpisodeStatistics` as above; the vector-level
  `gymnasium.wrappers.vector.RecordEpisodeStatistics` writes
  `infos["episode"]` instead and does not satisfy this.

With `vector_env_returns_torch: false` (default), the result is a numpy
vector env. The backend converts it to torch tensors on `device` and
translates Gymnasium's `final_obs` into `final_observation`, as on the
`gym.make` path.

With `vector_env_returns_torch: true`, the result must already return torch
tensors on `device` and follow rl-garden's env contract itself:
`reset(seed=...)`, Dict observations of tensors, and on episode end dense
`infos["final_observation"]`/`"final_info"`/`"_final_info"`. The backend uses it without conversion and only wraps it so
that `reset()` without a seed replays the configured seed. The adapter's
float32 space canonicalization is skipped too, so declare `float32` `Box`
spaces for float observations.

Frame stacking, `--reward_scale`/`--reward_bias` and `obs_key_map` apply to
the factory path exactly as to the `gym.make` path.

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

A backend that needs no knobs of its own can instead read an existing section
of the stock CLI by overriding `config_from_args`. It is then selectable with
`--env_backend <name>` from the stock entrypoints, configured through that
section, and the section counts as active for preflight validation:

```python
class MyIsaacLabTasksBackend(IsaacLabBackend):  # registered as "my_isaaclab_tasks"
    api_version = 2
    config_field = "my_isaaclab_tasks"

    @classmethod
    def config_from_args(cls, args):
        return args.isaaclab  # python examples/train_online.py ppo --env_backend my_isaaclab_tasks --isaaclab.<field> ...
```
