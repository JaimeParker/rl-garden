Tutorial learning path
======================

Start with :doc:`/start/quickstart` if you have not run an agent yet.
The chapters below build from configuration to your own environment and policy.
The chapters form a complete path; reference pages may still be drafts.

:doc:`01-configuration-and-cli` teaches you to read a preset, override a
field and inspect its source before construction. Use it whenever the effective
configuration differs from what you expected.

:doc:`02-online-rl` runs SAC and PPO on Pendulum and moves SAC to
ManiSkill PushCube. You locate metrics, run artifacts and a saved-policy
evaluation before increasing the training budget.

:doc:`03-observations-and-encoders` moves from state to RGB inputs,
separates actor and critic observations, and checks an encoder without a
renderer. It also explains the pretrained-weight cache and file formats.

:doc:`04-offline-pretraining` writes a synthetic H5 dataset, explains flat and
dictionary observations, and runs CPU BC and IQL before checkpoint evaluation.

:doc:`05-offline-to-online` carries a WSRL-compatible checkpoint into live
Pendulum, compares WSRL and Cal-QL presets, and separates configuration from replay handoff.

:doc:`06-checkpoints-and-evaluation` saves and resumes a Pendulum run
with replay, checks deterministic action equality after loading, and evaluates
through automatic algorithm selection. You can take it immediately after chapter 02.

:doc:`07-bring-your-own-environment` registers a one-dimensional Gymnasium
task from your module, trains SAC, and compares that path with the custom PointReach backend.

:doc:`08-bring-your-own-policy` trains a transformer PPO actor, a flow SAC
actor, an external image extractor, and a deterministic SAC actor. It checks
contracts and carries factory paths through checkpoint evaluation.

:doc:`09-extending-from-your-package` installs the downstream fixture and follows
its algorithm entry point, policy, rollout extras, loss, and checkpoint metadata.
The extension guide remains the contract for your own package.
