# Adding a New Algorithm

This document is the authoritative guide for adding a new RL algorithm to
rl-garden. Read it before touching `rl_garden/algorithms/`, any
`rl_garden/training/{online,offline,off2on}/` file, or any `_args.py`.

## Overview

An algorithm lives in two places:

```
rl_garden/algorithms/my_algo.py          ← learning logic (networks, update rule)
rl_garden/training/online/my_algo.py     ← CLI wiring, env construction, registration
```

The training file is auto-discovered and registered; no other file needs to be
changed.

---

## Part A — Algorithm implementation (`rl_garden/algorithms/`)

### Inheritance

Choose the appropriate base class:

| Base class | Use for | Abstract methods |
|---|---|---|
| `OffPolicyAlgorithm` | SAC, DDPG, and variants | `_setup_model()`, `train(gradient_steps, compute_info)` |
| `OnPolicyAlgorithm` | PPO and variants | `_setup_model()`, `update()` |
| `BaseAlgorithm` | custom loops | `_setup_model()`, `learn(total_timesteps)` |

`learn()` is already implemented on `OffPolicyAlgorithm` and `OnPolicyAlgorithm`;
only `BaseAlgorithm` subclasses must implement it directly.

### Minimal off-policy example

```python
# rl_garden/algorithms/my_algo.py
from rl_garden.algorithms.off_policy import OffPolicyAlgorithm

class MyAlgo(OffPolicyAlgorithm):

    def __init__(self, env, eval_env=None, *, my_lr: float = 3e-4, **kwargs):
        super().__init__(env, eval_env, **kwargs)
        self.my_lr = my_lr
        self._setup_model()

    def _setup_model(self) -> None:
        # Build networks, optimizers, replay buffer.
        # All tensors must live on self.device.
        self.policy = ...
        self.my_optimizer = ...

    def train(self, gradient_steps: int, compute_info: bool = False) -> dict[str, float]:
        # Sample from replay buffer and update networks.
        # Return a flat dict of scalar metrics for logging.
        batch = self.replay_buffer.sample(self.batch_size)
        ...
        return {"train/loss": loss.item()}

    # --- Checkpoint extension points (override as needed) ---

    def _optimizer_names(self) -> tuple[str, ...]:
        # List optimizer attributes to save/load automatically.
        return ("my_optimizer",)

    def _extra_checkpoint_state(self) -> dict:
        # Extra tensors or scalars to persist beyond policy + optimizers.
        return {"my_counter": self._my_counter}

    def _load_extra_checkpoint_state(self, state: dict) -> None:
        self._my_counter = state.get("my_counter", 0)

    def _training_state_dict(self) -> dict:
        # Loop-level counters (e.g. EMA weights, schedule step).
        return {}

    def _load_training_state_dict(self, state: dict) -> None:
        pass
```

**Rules:**
- `save()` and `load()` are already implemented on `BaseAlgorithm`; do not
  reimplement them. Extend via the hook methods above.
- If your algorithm can resume from a checkpoint saved by a different class
  (e.g. `OfflineSAC` resuming from an `SAC` checkpoint), set the class
  attribute `_compatible_checkpoint_algorithms = ("SAC",)`.
- All network tensors and replay buffers must stay on `self.device`.
  CPU-backed env observations are moved by `_obs_to_policy_device()` before
  inference; do not add ad-hoc `.cuda()` calls.
- Avoid NumPy in rollout / replay / update hot paths.

### Export the class

Add it to `rl_garden/algorithms/__init__.py`:

```python
from rl_garden.algorithms.my_algo import MyAlgo
```

---

## Part B — Training registration (`rl_garden/training/online/`)

Create `rl_garden/training/online/my_algo.py`. The file is auto-discovered
because `OnlineAlgorithmRegistry.discover()` imports every non-`_`-prefixed
module in the package. **No other file needs to be changed.**

The module must provide four components (in this order in the file):

### 1. `_my_algo_env_request(args, run_name) -> EnvRequest`

Translates `args` fields into a backend-neutral `EnvRequest`. Keep all
env-creation logic here; `build_my_algo` must not touch envs.

```python
def _my_algo_env_request(args, run_name):
    from rl_garden.common.cli_args import resolve_eval_record_dir
    from rl_garden.envs.backend_registry import EnvRequest

    is_visual = args.obs_mode != "state"
    backend_config = args.resolve_backend_config()
    eval_record_dir = resolve_eval_record_dir(args, run_name)
    return EnvRequest(
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        seed=args.seed,
        camera_width=args.camera_width if is_visual else None,
        camera_height=args.camera_height if is_visual else None,
        include_state=args.include_state if is_visual else True,
        per_camera_rgbd=args.per_camera_rgbd if is_visual else False,
        frame_stack=args.frame_stack,
        num_eval_envs=args.num_eval_envs,
        eval_record_dir=eval_record_dir,
        capture_video=args.capture_video,
        video_fps=args.video_fps,
        num_eval_steps=args.num_eval_steps,
        backend_config=backend_config,
    )
```

### 2. `build_my_algo(args, env, eval_env, logger, checkpoint_dir) -> MyAlgo`

Constructs the fully-initialised agent, including checkpoint loading.
All heavy imports (`MyAlgo`, encoders, etc.) go inside this function.

```python
def build_my_algo(args, env, eval_env, logger, checkpoint_dir):
    from rl_garden.algorithms.my_algo import MyAlgo

    agent = MyAlgo(
        env=env,
        eval_env=eval_env,
        my_lr=args.my_lr,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        seed=args.seed,
        logger=logger,
        std_log=args.std_log,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        num_eval_steps=args.num_eval_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    if args.load_checkpoint is not None:
        agent.load(args.load_checkpoint, load_replay_buffer=args.load_replay_buffer)
    return agent
```

### 3. `run_my_algo(args: MyAlgoArgs) -> None`

Calls `run_online`. Add any pre-flight guards before the call.

```python
def run_my_algo(args: MyAlgoArgs) -> None:
    from rl_garden.training.online._runner import run_online

    is_visual = args.obs_mode != "state"
    obs_tag = f"rgbd_{args.encoder}" if is_visual else "state"
    run_online(
        args,
        obs_tag=obs_tag,
        make_env_request=_my_algo_env_request,
        build_agent=build_my_algo,
        # post_learn=lambda agent: ...,  # optional cleanup after learn()
    )
```

### 4. Args dataclass + registration (at the bottom, after the functions)

```python
# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from rl_garden.common.env_args import EnvBackendArgs  # noqa: E402
from rl_garden.training.online._args import MyAlgoTrainingArgs  # noqa: E402
from rl_garden.training.online._registry import registry  # noqa: E402


@dataclass
class MyAlgoArgs(MyAlgoTrainingArgs, EnvBackendArgs):
    """MyAlgo — brief description.

    Env backend: ``--env_backend maniskill`` (default) or ``--env_backend robotwin``.
    """


registry.register("my_algo", MyAlgoArgs, run_my_algo)
```

The `# noqa: E402` comments suppress "module-level import not at top of file"
warnings. This pattern is intentional: the functions are defined before the
imports that trigger registration.

### `MyAlgoTrainingArgs` in `_args.py`

Algorithm hyperparameters belong in `rl_garden/training/online/_args.py`, not
inline in the run file (unless the algorithm is a one-off). Follow the double-
layer pattern when state-obs and visual-obs have different resource budgets:

```python
# rl_garden/training/online/_args.py

@dataclass
class MyAlgoTrainingArgs(EnvRunArgs, CheckpointArgs):
    """State-obs defaults."""
    total_timesteps: int = 1_000_000
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"
    batch_size: int = 1024
    my_lr: float = 3e-4
    gamma: float = 0.99

@dataclass
class VisionMyAlgoTrainingArgs(MyAlgoTrainingArgs, VisionArgs):
    """Visual-obs defaults — override state-obs values that need tighter budgets."""
    buffer_size: int = 200_000   # GPU memory constraint
    batch_size: int = 512
```

Use `VisionMyAlgoTrainingArgs` in the `MyAlgoArgs` inheritance chain when the
algorithm supports visual observations.

---

## Part C — Offline and Off2on algorithms

The same pattern applies; only the runner and registry differ:

| Phase | Runner | Registry |
|-------|--------|----------|
| online | `run_online(args, *, obs_tag, make_env_request, build_agent, post_learn)` | `rl_garden/training/online/_registry.py` |
| offline | `run_offline(args, *, build_agent)` | `rl_garden/training/offline/_registry.py` |
| off2on | phase-specific runner | `rl_garden/training/off2on/_registry.py` |

Offline run files live in `rl_garden/training/offline/`, follow the same
four-component structure, and are auto-discovered in the same way. The offline
runner receives only `(args, build_agent)` — no `make_env_request` — because
offline training typically uses a dataset rather than live rollouts.

---

## Verification

```bash
# Registry discovery:
python -c "
from rl_garden.training.online._registry import registry
registry.discover()
print(sorted(registry._entries))
"

# Config inspection (no env or agent created):
python examples/train_online.py my_algo --print-config 2>/dev/null | python -m json.tool | head -20

# Run tests (skip hardware-dependent tests):
pytest tests/ -q \
    --ignore=tests/test_maniskill_env.py \
    --ignore=tests/test_gpu_sac.py \
    --ignore=tests/test_act_provider.py
```
