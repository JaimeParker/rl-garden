<p align="center">
  <img src="docs/_static/logo.png" alt="rl-garden logo" width="360">
</p>

# rl-garden

`rl-garden` is a PyTorch-native robot-learning framework for online RL, offline RL,
imitation learning, and offline-to-online training. It provides reusable algorithm,
policy, encoder, replay-buffer, and environment-backend components for simulation,
offline datasets, and real-robot systems.

The framework keeps rollout, replay, inference, and update paths on torch tensors,
with GPU-vectorized execution as the preferred training path. Environment backends
are registered independently from algorithms so additional simulators and robot
platforms can be integrated without creating platform-specific training entrypoints.

## Capabilities

- Online reinforcement learning with SAC, PPO and their variants: [algorithms](docs/algorithms/index.rst).
- YAML policy graphs and decision-level PPO: [policy graphs](docs/guides/policy-graphs.md).
- Offline learning from datasets: [offline RL](docs/algorithms/offline-rl.rst).
- Imitation and generative policies: [imitation](docs/algorithms/imitation-and-generative.rst).
- Offline-to-online warm starts and fine-tuning: [offline-to-online](docs/algorithms/offline-to-online.rst).
- State, RGB, depth and asymmetric actor/critic observations: [observations and encoders](docs/tutorials/03-observations-and-encoders.rst).
- Tensor replay and explicit storage/sample devices: [core concepts](docs/start/concepts.rst).
- World models and planning: [model-based RL](docs/algorithms/model-based.rst).
- Independently registered simulator and dataset backends: [backends](docs/backends/index.rst).
- External actors, extractors and downstream packages: [bring your own policy](docs/tutorials/08-bring-your-own-policy.rst).
- Downstream robot integration and reward classifiers: [real-robot work](docs/real-robot.rst).
- Optional RLinf/Ray distributed training adapters: [RLinf integration](docs/design/rlinf-integration.md).

## Installation

```bash
pip install -e .
```

See [installation](docs/start/installation.rst) for extras and CUDA setup.

## Training

Train SAC on Pendulum on the CPU (the quickstart run):

```bash
rl-garden-train-online sac --env-backend gymnasium --env-id Pendulum-v1 \
  --num-envs 1 --num-eval-envs 1 --gymnasium.device cpu --buffer-device cpu \
  --total-timesteps 2000 --learning-starts 100 --training-freq 4 --batch-size 64 \
  --buffer-size 10000 --hidden-dim 64 --gamma 0.99 --eval-freq 0 --log-freq 500 \
  --no-capture-video --exp-name sac-pendulum
```

Pretrain IQL on the H5 dataset written by tutorial 04:

```bash
rl-garden-train-offline iql --offline-dataset runs/chapter04-data/pendulum.h5 \
  --dataset-backend h5 --env-backend gymnasium --env-id Pendulum-v1 \
  --device cpu --buffer-device cpu --buffer-size 256 --batch-size 16 \
  --action-low -2 --action-high 2 --n-critics 2 --critic-subsample-size 2 \
  --num-offline-steps 20 --eval-freq 0 --log-freq 10 --exp-name chapter04-iql
```

Preflight an offline-to-online WSRL preset without starting a simulator:

```bash
rl-garden-train-off2on wsrl --config configs/off2on/wsrl.yaml --print-config
```

Evaluate the quickstart checkpoint:

```bash
rl-garden-eval --checkpoint-path runs/sac-pendulum/checkpoints/final.pt \
  --device cpu --num-eval-envs 1 --num-eval-episodes 5
```

Every training command accepts `--print-config`, `--dry-run` and `--explain-param`; see the [CLI reference](docs/reference/cli.rst).

## Documentation

The [documentation site](https://jaimeparker.github.io/rl-garden/) is served by `.github/workflows/docs.yml` once Pages is enabled; the source entry point is [docs/index.rst](docs/index.rst).
The Diátaxis sections cover learning (Start and Tutorials), tasks (Algorithms and Backends), Reference and Design, with a Real robot pointer.

## Testing

`scripts/test.sh` provides fast, sim, gpu, slow, docs, linkcheck and all lanes.
Run the smallest relevant test file first.
See docs/README.md and .agents/runbooks/testing.md.

## Research Influences

The implementation combines ideas and engineering patterns from multiple projects
rather than treating any single framework as its template. Reference implementations
are kept under `3rd_party/`: `Cal-QL`, `wsrl`, and `implicit_q_learning` are real git
submodules (see `.agents/runbooks/baseline-install.md`); `CORL`,
`NexRL`, `RL-100`, `RLinf`, `dppo`, `fql`, `hil-serl`, `qam`, `qc`, `qgf`, and
`stable-baselines3` are untracked read-only reference clones. Treat all of these
directories as read-only unless a change is explicitly requested.
