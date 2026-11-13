# Documentation development

Documentation is built with Sphinx. The entry point is [index.rst](index.rst).
Run these commands from the repository root:

```bash
pip install -r docs/requirements.txt
scripts/test.sh docs
scripts/test.sh linkcheck
python tools/docs/gen_tables.py --check
python -m pytest tests/test_docs_generated.py tests/test_tutorial_commands.py tests/test_tutorial_examples.py -q -p no:cacheprovider
```

Start with the smallest relevant test file; `scripts/test.sh all` runs fast, sim, gpu, slow and docs lanes.
Backend and accelerator checks require their optional runtime and hardware; report unavailable dependencies instead of changing execution semantics.
See `.agents/runbooks/testing.md` for the testing procedure.

## Anti-rot rules

- Pull Python examples in with `literalinclude` and paired `# docs: start <name>` / `# docs: end <name>` markers.
- Execute each tutorial example through `tests/test_tutorial_examples.py` with a small CPU smoke budget.
- Keep every documented training/evaluation argv in `tests/test_tutorial_commands.py`; training uses static preflight validation.
- Regenerate registry tables with `python tools/docs/gen_tables.py` and verify byte equality with `--check` and `tests/test_docs_generated.py`.
- Build with `scripts/test.sh docs`, which enforces Sphinx `-W -n --keep-going`.

## Final tree

```text
docs/README.md
docs/REFERENCES.md
docs/algorithms/imitation-and-generative.rst
docs/algorithms/index.rst
docs/algorithms/model-based.rst
docs/algorithms/offline-rl.rst
docs/algorithms/offline-to-online.rst
docs/algorithms/ppo-family.rst
docs/algorithms/sac-family.rst
docs/backends/custom.rst
docs/backends/index.rst
docs/backends/maniskill.rst
docs/backends/minari.rst
docs/backends/mujoco-warp.rst
docs/backends/mujoco.rst
docs/design/index.rst
docs/design/iql-overview.md
docs/design/rlinf-integration.md
docs/design/rng-numerical-determinism.md
docs/design/wsrl-overview.md
docs/guides/checkpoint.md
docs/guides/configuration.md
docs/guides/d4rl-legacy-expansion.md
docs/guides/decision-level-ppo.md
docs/guides/dm-control-integration.md
docs/guides/dreamer-v3.md
docs/guides/extending.md
docs/guides/gymnasium-custom-envs.md
docs/guides/isaaclab-camera-stall.md
docs/guides/isaaclab-custom-tasks.md
docs/guides/metaworld-integration.md
docs/guides/ncrl.md
docs/guides/offline-acceleration.md
docs/guides/ogbench-integration.md
docs/guides/policy-graphs.md
docs/guides/rlbench-integration.md
docs/guides/robomimic-integration.md
docs/guides/robotwin.md
docs/guides/supermimic.md
docs/guides/wsrl-reproduction.md
docs/index.rst
docs/real-robot.rst
docs/reference/_generated/algorithms.rst
docs/reference/_generated/backends.rst
docs/reference/checkpoints.rst
docs/reference/cli.rst
docs/reference/configuration.rst
docs/reference/index.rst
docs/reference/python-api.rst
docs/reference/references.rst
docs/start/concepts.rst
docs/start/installation.rst
docs/start/quickstart.rst
docs/tutorials/01-configuration-and-cli.rst
docs/tutorials/02-online-rl.rst
docs/tutorials/03-observations-and-encoders.rst
docs/tutorials/04-offline-pretraining.rst
docs/tutorials/05-offline-to-online.rst
docs/tutorials/06-checkpoints-and-evaluation.rst
docs/tutorials/07-bring-your-own-environment.rst
docs/tutorials/08-bring-your-own-policy.rst
docs/tutorials/09-extending-from-your-package.rst
docs/tutorials/index.rst
```
