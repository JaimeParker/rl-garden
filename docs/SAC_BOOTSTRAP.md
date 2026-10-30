# SAC bootstrapping at episode boundaries

SAC and Residual SAC use `--bootstrap-at-done` to control whether the target includes the estimated value of the next state.

| Mode | Continuing episode | Terminated | Time-limit truncation only | Both flags |
| --- | --- | --- | --- | --- |
| `always` | Bootstrap | Bootstrap | Bootstrap | Bootstrap |
| `never` | Bootstrap | Stop | Stop | Stop |
| `truncated` (default, recommended) | Bootstrap | Stop | Bootstrap | Stop |

Stopping bootstrap leaves the immediate reward as the target. Otherwise, the target also includes the discounted next-state value and the SAC entropy term.

```bash
python examples/train_online.py residual_sac --bootstrap-at-done truncated
```

Supply the environment and base-model arguments as appropriate. Online arguments, algorithm constructors and the residual H5 loader default to `truncated`; explicit `always` and `never` remain available. Preserve the saved configuration when reproducing or continuing older experiments.

## Experimental motivation

In a subset of open_laptop visual and privileged-state residual comparisons, `always` produced persistent Q-value oscillations and abrupt increases or decreases, including late in training. Some runs retained critic losses on the order of 1–10, with occasional values above 10. Corresponding `truncated` runs were generally smoother after the initial adjustment, with late critic losses mostly below 0.2 in the inspected comparisons. These observations motivate the recommended default.

With `always`, a successful terminal state's predicted value still contributes to the target even though no subsequent transition is collected from that state. This can amplify value-estimation errors. With `truncated`, true termination removes this contribution, while time limits retain it. The curves support this explanation but do not establish it as the only cause.

These ranges describe selected comparisons, not every configuration. Critic loss depends on reward scaling and the loss definition. Smoother Q estimates and lower loss do not guarantee higher task success; compare policies using the same evaluation scenes and execution settings.

## Final observations and replay

An automatic reset may return the next episode's initial observation. For ended environments, the transition uses the previous episode's final observation without overwriting observations from continuing environments. Replay `dones` indicate whether bootstrap stops, which is not necessarily the raw environment termination flag.

Changing the option does not relabel existing replay. Parameter comparisons should use independent residual runs with the same base and warmup policies.

H5 data with explicit termination and truncation flags support the same modes. Legacy data containing only `done` retain that flag because the two causes cannot be distinguished.
