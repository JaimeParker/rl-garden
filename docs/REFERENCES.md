# Algorithm References

`rl-garden` reimplements published RL/IL algorithms in a shared PyTorch framework.
This page lists, for every registered algorithm, the paper it implements and the
reference implementation(s) it was ported from or cross-checked against. Sources
are taken from each algorithm's own module docstring
(`rl_garden/algorithms/<name>.py`); see that file for exact line-level formula
citations where the port involved a non-obvious translation.

Three categories appear below:

- **Ported** — code structure/formulas translated directly from a named external
  reference implementation.
- **Reimplemented from paper** — rl-garden's own implementation of a published
  algorithm, written from the paper and cross-checked against one or more
  external references (no single reference was translated line-by-line).
- **rl-garden extension** — new capability (action chunking, recurrence,
  DDP, off2on wiring, ...) layered on top of an already-listed algorithm; no
  separate external port.

External reference repositories are vendored read-only under `3rd_party/` (see
`.gitmodules` and each subdirectory's own `README`/`LICENSE`) except where noted
as "not vendored" — those were consulted during development but are not part of
this repository.

## Online RL

| Algorithm | Module | Paper | Reference implementation |
|---|---|---|---|
| SAC | `algorithms/sac.py` | Haarnoja et al. 2018, [arXiv:1801.01290](https://arxiv.org/abs/1801.01290) | Reimplemented from paper, rlpd, and [wsrl](https://github.com/zhouzypaul/wsrl); structural template: ManiSkill's `examples/baselines/sac/sac.py` |
| PPO | `algorithms/ppo.py` | Schulman et al. 2017, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | Reimplemented from paper; style/structure: [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) + ManiSkill baselines |
| DDPG (DrQ-v2) | `algorithms/ddpg.py` | Lillicrap et al. 2015 (DDPG, [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)) + Yarats et al. 2021 (DrQ-v2, [arXiv:2107.09645](https://arxiv.org/abs/2107.09645)) | Ported from [drqv2](https://github.com/facebookresearch/drqv2) (not currently vendored) |
| TD3 | `algorithms/td3.py` | Fujimoto et al. 2018, [arXiv:1802.09477](https://arxiv.org/abs/1802.09477) | Reimplemented from paper on top of rl-garden's own `DDPG`/`DrQv2Critic` |
| FlashSAC | `algorithms/flash_sac.py` | [arXiv:2409.08689](https://arxiv.org/abs/2409.08689) | Ported from [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC) (not vendored) |
| RLPD | `algorithms/rlpd.py` | Ball et al. 2023, [arXiv:2302.02948](https://arxiv.org/abs/2302.02948) | Reimplemented from paper on top of rl-garden's own `SAC` |
| RLPDHybrid | `algorithms/rlpd_hybrid.py` | RLPD (above) + HIL-SERL's hybrid discrete-gripper head, Luo et al. 2024, [arXiv:2410.21845](https://arxiv.org/abs/2410.21845) | Ported (hybrid head) from [hil-serl](https://github.com/rail-berkeley/hil-serl) |
| ExPLORe | `algorithms/explore.py` | Li et al. 2023, [arXiv:2311.05067](https://arxiv.org/abs/2311.05067) | Built on rl-garden's own `RLPD`; reward/mask relabeling head and RND novelty bonus ported from [ExPLORe](https://github.com/facebookresearch/ExPLORe) |
| SUPE | `algorithms/supe.py` | Wilcoxson et al. 2025, [arXiv:2410.18076](https://arxiv.org/abs/2410.18076) | Built on rl-garden's own `ExPLORe` (inherited verbatim: reward/mask relabeling, RND, offline/online mixing); skill-macro-action env wrapper and offline skill-relabeling ported from [SUPE](https://github.com/rail-berkeley/SUPE); consumes an `OPAL` (below) checkpoint for its frozen skill decoder |
| TDMPC2 | `algorithms/tdmpc2/` | Hansen et al. 2023, [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) | Ported from [tdmpc2](https://github.com/nicklashansen/tdmpc2) (not currently vendored) |
| DPPO | `algorithms/dppo.py` | Ren et al. 2024, [arXiv:2409.00588](https://arxiv.org/abs/2409.00588) | Ported from [dppo](https://github.com/irom-princeton/dppo) |
| SACFlow | `algorithms/sac_flow.py` | RLinf, [arXiv:2509.15965](https://arxiv.org/abs/2509.15965) | Ported from [RLinf](https://github.com/RLinf/RLinf) |
| ACRLPD | `algorithms/acrlpd.py` | Q-chunking, Li et al. 2025, [arXiv:2507.07969](https://arxiv.org/abs/2507.07969) | Ported from [QC](https://github.com/ColinQiyangLi/qc) |
| RecurrentSAC / TransformerSAC / RecurrentPPO / TransformerPPO | `algorithms/{recurrent,transformer}_{sac,ppo}.py`, `sequence_{sac,ppo}.py` | rl-garden extension | Techniques drawn on: R2D2 (Kapturowski et al. 2019, ICLR — no arXiv preprint) for recurrent replay/burn-in; GTrXL (Parisotto et al. 2019, [arXiv:1910.06764](https://arxiv.org/abs/1910.06764)) for the transformer variant |
| SACDDP | `algorithms/sac_ddp.py` | rl-garden extension | Single-node multi-GPU DDP wrapper around `SAC`; no external port |
| DAgger | `algorithms/dagger.py` | Ross, Gordon & Bagnell 2011, AISTATS (no arXiv preprint) | rl-garden-native; compared during design against [imitation](https://github.com/HumanCompatibleAI/imitation) |
| PolicyDistillation | `algorithms/policy_distillation.py` | Teacher-student distillation pattern used in legged-robot sim-to-real RL | Ported from [rsl_rl](https://github.com/leggedrobotics/rsl_rl) |

## Offline RL and Imitation Learning

| Algorithm | Module | Paper | Reference implementation |
|---|---|---|---|
| BC | `algorithms/bc.py` | Canonical behavioral cloning (no single source paper) | rl-garden-native |
| FlowBC | `algorithms/flow_bc.py` | Flow matching: Lipman et al. 2023, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | rl-garden-native (hand-rolled CondOT loss, no external flow-matching library dependency) |
| DiffusionBC | `algorithms/diffusion_bc.py` | Diffusion Policy, Chi et al. 2023, [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | Ported from [dppo](https://github.com/irom-princeton/dppo) |
| VisionDiffusionBC | `algorithms/vision_diffusion_bc.py` | Same as DiffusionBC, vision-conditioned | Sibling of `DiffusionBC`; vision-conditioning precedent also present locally at [diffusion_policy](https://github.com/real-stanford/diffusion_policy) |
| A2ABC | `algorithms/a2a_bc.py` | A2A (Action-to-Action flow matching), [arXiv:2602.07322](https://arxiv.org/abs/2602.07322) | Ported from [A2A_Flow_Matching](https://github.com/JIAjindou/A2A_Flow_Matching) |
| BCQ | `algorithms/bcq.py` | Fujimoto et al. 2019, [arXiv:1812.02900](https://arxiv.org/abs/1812.02900) | Ported from `sfujim/BCQ/continuous_BCQ/BCQ.py` (official reference, not vendored) |
| PLAS | `algorithms/plas.py` | Zhou et al. 2020, [arXiv:2011.07213](https://arxiv.org/abs/2011.07213) | Ported from `Wenxuan-Zhou/PLAS/algos.py` (official reference, not vendored) |
| IQL | `algorithms/iql.py` | Kostrikov et al. 2021, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) | Reimplemented from paper; cross-checked against [CORL](https://github.com/tinkoff-ai/CORL), [wsrl](https://github.com/zhouzypaul/wsrl), and the official JAX repo [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning) |
| IDQL | `algorithms/idql.py` | Hansen-Estruch et al. 2023, [arXiv:2304.10573](https://arxiv.org/abs/2304.10573) | Value/critic training reimplemented matching IQL's own formulas (rl-garden's `IQL` is a sibling, not a parent); diffusion actor built on rl-garden's own `DiffusionMLP`/`DiffusionProcess`; actor-extraction procedure ported from [IDQL](https://github.com/philippe-eecs/IDQL) |
| HILP | `algorithms/hilp.py` | Park et al. 2024, [arXiv:2402.15567](https://arxiv.org/abs/2402.15567) | Ported from [HILP](https://github.com/seohongpark/HILP) |
| OPAL | `algorithms/opal.py` | Ajay et al. 2021, [arXiv:2010.13611](https://arxiv.org/abs/2010.13611) | VAE (BiGRU posterior encoder, obs-conditioned prior, Gaussian decoder) ported from [SUPE](https://github.com/rail-berkeley/SUPE)'s own OPAL pretraining code; decoder reuses rl-garden's own `UnsquashedGaussianActor` |
| CQL | `algorithms/cql.py` | Kumar et al. 2020, [arXiv:2006.04779](https://arxiv.org/abs/2006.04779) | Reimplemented from paper; cross-checked against [CORL](https://github.com/tinkoff-ai/CORL), [wsrl](https://github.com/zhouzypaul/wsrl), and [Cal-QL](https://github.com/nakamotoo/Cal-QL) |
| CalQL | `algorithms/calql.py` | Nakamoto et al. 2023, [arXiv:2303.05479](https://arxiv.org/abs/2303.05479) | Extends rl-garden's own `CQL`; cross-checked against [Cal-QL](https://github.com/nakamotoo/Cal-QL) and [CORL](https://github.com/tinkoff-ai/CORL) |
| EDAC | `algorithms/edac.py` | An et al. 2021 (SAC-N / EDAC), [arXiv:2110.01548](https://arxiv.org/abs/2110.01548) | Ported from [CORL](https://github.com/tinkoff-ai/CORL) |
| SPOT | `algorithms/spot.py` | Wu et al. 2022, [arXiv:2202.06239](https://arxiv.org/abs/2202.06239) | Ported from [CORL](https://github.com/tinkoff-ai/CORL) |
| ReBRAC | `algorithms/rebrac.py` | Tarasov et al. 2023, [arXiv:2305.09836](https://arxiv.org/abs/2305.09836) | Ported from [CORL](https://github.com/tinkoff-ai/CORL) |
| TD3BC | `algorithms/td3_bc.py` | Fujimoto & Gu 2021, [arXiv:2106.06860](https://arxiv.org/abs/2106.06860) | Ported from [CORL](https://github.com/tinkoff-ai/CORL) |
| AWAC | `algorithms/awac.py` | Nair et al. 2020, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359) | Ported from [CORL](https://github.com/tinkoff-ai/CORL) |
| FQL | `algorithms/fql.py` | Park et al. 2025, [arXiv:2502.02538](https://arxiv.org/abs/2502.02538) | Ported from [fql](https://github.com/seohongpark/fql) |
| QGF | `algorithms/qgf.py` | [arXiv:2606.11087](https://arxiv.org/abs/2606.11087) | Ported from [qgf](https://github.com/zhouzypaul/qgf) |
| QAM | `algorithms/qam.py` | [arXiv:2601.14234](https://arxiv.org/abs/2601.14234), building on Adjoint Matching (Domingo-Enrich et al. 2024, [arXiv:2409.08861](https://arxiv.org/abs/2409.08861)) | Ported from [qgf](https://github.com/zhouzypaul/qgf) and standalone [qam](https://github.com/ColinQiyangLi/qam) |
| TDMPC2 (multitask) | `algorithms/tdmpc2/multitask/` | Same as online TDMPC2 | Same as online TDMPC2 |

## Offline-to-Online

| Algorithm | Module | Paper | Reference implementation |
|---|---|---|---|
| WSRL | `algorithms/wsrl.py` | Zhou et al. 2024, [arXiv:2412.07762](https://arxiv.org/abs/2412.07762) | Built on rl-garden's own `CalQL`; cross-checked against [wsrl](https://github.com/zhouzypaul/wsrl) |
| Off2OnCalQL | `algorithms/off2on_calql.py` | Nakamoto et al. 2023, [arXiv:2303.05479](https://arxiv.org/abs/2303.05479) | Cal-QL's own off2on design; cross-checked against [Cal-QL](https://github.com/nakamotoo/Cal-QL) |
| Off2OnIQL | `algorithms/off2on_iql.py` | Kostrikov et al. 2021, [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) | Off2on switch behavior confirmed against [wsrl](https://github.com/zhouzypaul/wsrl) |
| Off2OnAWAC | `algorithms/off2on_awac.py` | Nair et al. 2020, [arXiv:2006.09359](https://arxiv.org/abs/2006.09359) | Built on rl-garden's own `AWAC` |
| Off2OnSPOT | `algorithms/off2on_spot.py` | Wu et al. 2022, [arXiv:2202.06239](https://arxiv.org/abs/2202.06239) | Off2on switch behavior confirmed against [CORL](https://github.com/tinkoff-ai/CORL) |
| ACFQL | `algorithms/acfql.py` | Q-chunking, Li et al. 2025, [arXiv:2507.07969](https://arxiv.org/abs/2507.07969) + FQL, Park et al. 2025, [arXiv:2502.02538](https://arxiv.org/abs/2502.02538) | Ported from [QC](https://github.com/ColinQiyangLi/qc) |

## Shared infrastructure (not algorithm-specific)

The following are not separate algorithms but are cited in multiple algorithms'
docstrings above and are rl-garden's own framework code, not ports:
`OffPolicyAlgorithm`/`OnPolicyAlgorithm`/`OfflineRLAlgorithm` (base training
loops), `Off2OnReplayMixin` (generic offline→online transition machinery),
`ChunkedTensorReplayBuffer`/`_chunked_rollout.py` (generalizes [QC](https://github.com/ColinQiyangLi/qc)'s
single-env action-chunk queue to GPU-batched, per-env-staggered rollout).
