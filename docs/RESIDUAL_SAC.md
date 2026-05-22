# ResidualSAC 设计说明

本文档说明当前新增的 ResidualSAC 继承结构、动作坐标约定、功能边界，以及为支持 residual 训练对原有 SAC 代码做的最小改动。

## 继承结构

当前只实现 SAC residual，不包含 WSRL/CQL residual。

```text
ResidualSAC
└── SAC
    ├── SACCore
    └── OffPolicyAlgorithm
        └── BaseAlgorithm
```

相关类：

- `ResidualSAC`：位于 `rl_garden/algorithms/residual.py`，继承 `SAC`。
- `ResidualSACPolicy`：位于 `rl_garden/policies/residual_policy.py`，继承 `SACPolicy`。
- `ResidualTensorReplayBuffer`：继承 `TensorReplayBuffer`。
- `ResidualDictReplayBuffer`：继承 `DictReplayBuffer`。
- `ActionScaler`：位于 `rl_garden/common/action_scaler.py`，负责 env action 和 normalized action 的互相转换。
- `ACTBaseActionProvider`：位于 `rl_garden/models/act/provider.py`，把 ACT checkpoint 包装成 ResidualSAC 的 base-action provider。

也就是说，ResidualSAC 尽量复用 SAC 的初始化、优化器、alpha tuning、scheduler、target network update、logging 和 checkpoint 逻辑，只覆盖 residual 特有的 action 计算和 replay buffer 字段。

## 动作坐标约定

ResidualSAC 按 resfit 的方式处理 action：

- 算法内部、replay buffer、critic 输入都使用 normalized action，范围是 `[-1, 1]`。
- base policy 输出 raw env action。
- `ActionScaler.scale()` 把 raw env action 转成 normalized base action。
- `ActionScaler.unscale()` 把 final normalized action 转回 raw env action，送给 `env.step()`。

一次 rollout 的动作流是：

```text
base_action_raw = base_action_provider(obs)
base_naction = action_scaler.scale(base_action_raw)

unit_residual = actor(obs, base_naction)              # [-1, 1]
residual_naction = unit_residual * residual_action_scale
final_naction = clamp(base_naction + residual_naction, -1, 1)

env_action = action_scaler.unscale(final_naction)
env.step(env_action)
```

replay buffer 中：

- `actions` 存 `final_naction`，即真正执行的 normalized final action。
- `base_actions` 存当前步 normalized base action。
- `next_base_actions` 存下一步 normalized base action。

注意：`actions` 不是 residual delta，也不是 raw env action。

## Update 逻辑

ResidualSAC 的 critic 当前项仍然使用 replay 中的 final action：

```text
Q(s_t, final_naction_t)
```

target Q 使用下一步 base action 和 actor 预测 residual 组合：

```text
Q(s_{t+1}, clamp(base_naction_{t+1} + residual_actor(s_{t+1}, base_naction_{t+1}), -1, 1))
```

actor loss 使用当前步 base action 和 actor 预测 residual 组合：

```text
Q(s_t, clamp(base_naction_t + residual_actor(s_t, base_naction_t), -1, 1))
```

entropy/log-prob 仍按 actor 原生输出的 unit residual action 分布计算，不对 `residual_action_scale` 做额外 log-prob 修正。

## 新增功能

### Zero base debug provider

`examples/train_residual_sac_rgbd.py` 和 `examples/train_residual_sac_rgbd_peg.py`
都支持 `--debug`。debug 模式下使用 `ZeroBaseActionProvider`，它始终返回 raw
env action 空间里的全 0 动作，用于测试 residual rollout/update，不依赖 ACT
checkpoint。

### ACT base provider

非 debug 模式默认使用 ACT：

```bash
scripts/train_residual_sac_rgbd_peg.sh \
  --policy act \
  --ckpt-path act-peg-only
```

`--ckpt-path` 可以传完整 `.pt` 路径，也可以传名字。名字会按
`pretrained_models/<name>.pt` 解析，默认 `act-peg-only` 对应
`pretrained_models/act-peg-only.pt`。也可以用 `$RL_GARDEN_PRETRAINED_DIR`
覆盖 pretrained 根目录。

`act-peg-only.pt` 是 state-only ACT checkpoint，期望 43 维 state 和 6 维
env action。peg RGB residual 训练的 actor/critic 仍然用 RGB dict observation；
ACT base provider 会在需要 base action 时从 peg env 的 `base_env` 重建这 43
维 full state，避免把 residual actor 的观测格式改成 state-only。

ACT provider 的输出是 raw env action。后续仍由 ResidualSAC 的 `ActionScaler`
转换成 normalized base action。

### 训练入口

新增 shell launchers：

```bash
scripts/train_residual_sac_rgbd.sh       # generic ManiSkill RGBD residual SAC
scripts/train_residual_sac_rgbd_peg.sh   # PegInsertionSidePegOnly-v1 defaults
scripts/train_residual_sac_state_peg.sh  # PegInsertionSidePegOnly-v1 state obs
```

generic debug 启动方式：

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_rgbd.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --debug \
  --log_type tensorboard
```

peg-only debug 启动方式：

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_rgbd_peg.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --debug \
  --log_type tensorboard
```

peg-only state obs 启动方式：

```bash
CUDA_VISIBLE_DEVICES=2 scripts/train_residual_sac_state_peg.sh \
  --control_mode pd_ee_twist \
  --residual-action-scale 1 \
  --policy act \
  --ckpt-path act-peg-only \
  --log_type tensorboard
```

其中：

- `--debug`：使用全 0 base action provider。
- `--policy act`：使用 ACT 作为 base policy；`--debug` 会覆盖成 zero provider。
- `--ckpt-path act-peg-only`：加载 `pretrained_models/act-peg-only.pt`。
- `--residual-action-scale 1`：actor 输出的 unit residual 不缩小，residual normalized delta 范围为 `[-1, 1]`。
- `--control_mode pd_ee_twist`：使用 EE twist 控制模式。
- `--log_type tensorboard`：写 TensorBoard 日志。

`scripts/train_residual_sac.sh` 保留为兼容别名，转发到 generic
`scripts/train_residual_sac_rgbd.sh`。generic 入口参考 `train_sac_rgbd.py`，
不会传 peg-only kwargs；peg-only 的 `fix_peg_pose`、`fix_box`、`robot_uids`
和 `reward_mode` 只在 peg residual entrypoints 中出现。

## 对原有代码的改动

为了让 ResidualSAC 复用 SACCore 和 OffPolicyAlgorithm，而不是复制整套 SAC 训练循环，原有代码增加了几组 no-op hook。普通 SAC 默认行为不变。

### `rl_garden/algorithms/sac.py`

新增构造 hook：

- `_policy_action_space()`
- `_build_policy(features_extractor)`

普通 SAC 返回 env action space 和 `SACPolicy`。ResidualSAC 覆盖这两个 hook，使用 normalized residual action space 和 `ResidualSACPolicy`。

### `rl_garden/algorithms/sac_core.py`

新增 update hook：

- `_actor_action_log_prob(obs, base_actions=None, stop_gradient=False)`
- `_target_action_log_prob(data)`
- `_actor_loss_from_batch(data)`

普通 SAC 忽略 `base_actions`。ResidualSAC 覆盖这些 hook，在 target Q 和 actor loss 中使用 `base + residual`。

`_slice_batch()` 也增加了对 `base_actions` / `next_base_actions` 的可选透传，保证 high-UTD 分 batch 时 residual 字段不会丢失。

### `rl_garden/algorithms/off_policy.py`

新增 rollout hook：

- `_on_env_reset(obs)`
- `_rollout_action(obs, learning_has_started)`
- `_replay_buffer_add_kwargs(...)`
- `_post_rollout_step(...)`

普通 SAC 的默认逻辑是 `replay_action == env_action`。ResidualSAC 覆盖后变成：

- replay 存 normalized final action。
- env 接收 unscaled raw action。
- replay 额外写入 base action 字段。

### `rl_garden/common/checkpoint.py`

replay buffer checkpoint 支持可选字段：

- `base_actions`
- `next_base_actions`

普通 replay buffer 没有这些字段时，checkpoint 格式不变。

### 导出入口

新增导出：

- `rl_garden.algorithms.ResidualSAC`
- `rl_garden.policies.ResidualSACPolicy`
- `rl_garden.buffers.ResidualTensorReplayBuffer`
- `rl_garden.buffers.ResidualDictReplayBuffer`
- `rl_garden.common.ActionScaler`
- `rl_garden.models.act.ACTBaseActionProvider`

## 测试

新增 `tests/test_residual_sac.py`，覆盖：

- `ActionScaler` raw/normalized 互转。
- residual tensor/dict replay buffer 的 base action 字段。
- rollout 中 env 接收 raw action、replay 存 normalized action。
- update hook 中 target Q 和 actor loss 使用 `base + residual`。

短 smoke：

```bash
scripts/train_residual_sac_rgbd_peg.sh \
  --policy act \
  --ckpt-path act-peg-only \
  --control_mode pd_ee_twist \
  --log_type none \
  --no_std_log \
  --num_envs 1 \
  --num_eval_envs 1 \
  --total_timesteps 8 \
  --learning_starts 4 \
  --training_freq 4 \
  --batch_size 2 \
  --buffer_size 64 \
  --buffer_device cpu \
  --sim_backend physx_cpu \
  --render_backend gpu \
  --eval_freq 0 \
  --log_freq 4
```
