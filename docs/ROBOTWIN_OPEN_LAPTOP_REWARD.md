# RoboTwin `open_laptop` Reward

## Summary

This document describes the configurable dense/hybrid reward implementation for
RoboTwin `open_laptop`, introduced in the
[reward implementation commit](https://github.com/Nole326/rl-garden/commit/ef29090e5f5e7dc0406cb26d7f584a195f41ba24).
For the complete training pipeline, see
[Residual SAC Pipeline](ROBOTWIN_OPEN_LAPTOP_RESIDUAL_CURRENT.md).

The reward implementation includes:

- Add the `OpenArticulation` reward primitive.
- Register `open_laptop` in the RoboTwin reward registry.
- Expose and validate configurable hybrid reward shaping controls.
- Forward those controls through the RoboTwin backend.
- Add focused tests for registry coverage, shaping math, and backend forwarding.
- Keep legacy RoboTwin reward factory tests compatible with tasks that do not
  implement `check_success()`.

## Scope and compatibility

This document covers reward construction and configuration forwarding only.
Encoder, executor, and other training components are described in the pipeline
document linked above. Adding this documentation does not change runtime code,
training parameters, or the reward implementation.

The default reward behavior remains unchanged: `reward_shaping_mode` still
defaults to `absolute`. The hybrid formula below is used only when
`reward_shaping_mode="hybrid"` is selected explicitly.

## Reward formulation

For `open_laptop`, the `OpenArticulation` primitive first computes a raw
articulation score from reaching the laptop handle and opening the target joint:

$$
s_t = 1 - \tanh(5d_t) + 3\,\mathrm{progress}_t
$$

Here $d_t$ is the distance from the selected robot TCP to the laptop contact
point. The selected TCP is the configured arm when `arm_tag` is provided;
otherwise it is the closer TCP.

The target joint position is:

$$
q^{\star} = lower + 0.4(upper - lower)
$$

Let $q_0$ be the joint position when the reward primitive is initialized. The
opening progress is:

If $q_0 \lt q^{\star}$, progress is the clipped normalized joint advancement:

$$
\mathrm{progress}_t = \mathrm{clip}\left(\dfrac{q_t - q_0}{q^{\star} - q_0}, 0, 1\right)
$$

If $q_0 \ge q^{\star}$, progress is an indicator that the joint remains at or
beyond the target:

$$
\mathrm{progress}_t = \mathbf{1}[q_t \ge q^{\star}]
$$

Because this primitive has `max_reward=4` and the top-level `SerialTask`
normalizes rewards by default, the adapter receives the dense potential:

$$
p_t = \frac{s_t}{4}
$$

If the reward tree has already advanced to the terminal `Success` subtask, the
normalized dense potential is $p_t=1$. If the reward object reports failure,
`_dense_potential()` returns $0$.

The hybrid reward then uses the dense-potential delta:

$$
\Delta p_t = p_t - p_{t-1}
$$

$$
r_{\mathrm{raw}} = 0.03p_t + 3.0\Delta p_t + r_{\mathrm{success}} + r_{\mathrm{step}} + r_{\mathrm{stall}} + r_{\mathrm{backtrack}}
$$

The final environment reward applies the configured affine transform:

$$
r_t = \mathrm{reward\_scale}\,r_{\mathrm{raw}} + \mathrm{reward\_bias}
$$

For the current hybrid configuration, `reward_scale=1.0` and
`reward_bias=0.0`.

Reward components are:

- On success, $r_{\mathrm{success}}=10.0$. This success reward is added to the
  dense and delta terms; it does not override them.
- Successful steps do not subtract $r_{\mathrm{step}}$,
  $r_{\mathrm{stall}}$, or $r_{\mathrm{backtrack}}$.
- When not successful, $r_{\mathrm{step}}=-0.003$.
- When not successful and $|\Delta p_t|<10^{-4}$,
  $r_{\mathrm{stall}}=-0.035$.
- When not successful and $\Delta p_t<-10^{-4}$,
  $r_{\mathrm{backtrack}}=-0.06$.

## Current hybrid reward settings

The current hybrid configuration uses:

- `dense_success_reward=10.0`
- `dense_weight=0.03`
- `relative_weight=3.0`
- `step_penalty=0.003`
- `stall_threshold=0.0001`
- `stall_penalty=0.035`
- `backtrack_penalty=0.06`
- `reward_scale=1.0`
- `reward_bias=0.0`

`potential_discount=0.99` and `potential_weight=5.0` are used by the separate
`potential` shaping mode; they do not participate in the current hybrid
formula.

## How to enable

Use the existing dense reward path and explicitly select hybrid shaping:

- `reward_mode="dense"`
- `reward_shaping_mode="hybrid"`
- set the hybrid parameters listed above as needed

Leaving `reward_shaping_mode` unset keeps the default `absolute` behavior.

## Historical validation

The following checks were recorded for the original reward submission on
September 3, 2026; they are not new simulation results from this documentation
update.

- Target branch before the reward change:
  `Nole326/rl-garden:dev/residual-robotwin-delta-ee@a7987460cd51105d5e7d68d175a207e8f8bcfb57`.
- In an isolated Python 3.10 environment:
  - `git diff --check` on the 8 PR files: passed.
  - `python3 -m py_compile` on the 8 PR files: passed.
  - focused test collection: 12 tests collected.
  - focused reward tests: 7 passed.
  - selected safe RoboTwin compatibility tests: 5 passed.

## Implementation files

- [`reward.py`](../rl_garden/envs/robotwin/reward.py): defines `OpenArticulation` and reward composition primitives.
- [`rewards/registry.py`](../rl_garden/envs/robotwin/rewards/registry.py): registers the `open_laptop` reward tree.
- [`adapter.py`](../rl_garden/envs/robotwin/adapter.py): computes the dense potential and shaping components.
- [`config.py`](../rl_garden/envs/robotwin/config.py): stores and validates reward settings.
- [`env_args.py`](../rl_garden/common/env_args.py) and the [RoboTwin backend](../rl_garden/envs/backends/robotwin.py): expose and forward those settings.
- [Focused reward tests](../tests/test_robotwin_b10c_reward.py) and [environment compatibility tests](../tests/test_robotwin_env.py): cover reward math, registration, configuration forwarding, and compatibility.
