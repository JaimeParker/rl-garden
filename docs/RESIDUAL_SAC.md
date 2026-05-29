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
- `BasePolicyProvider`：位于 `rl_garden/policies/base_policies/base.py`，所有 base policy 的抽象基类（`nn.Module` 子类），定义 `select_action() → BasePolicyOutput`、`reset()`、`bind_env()` 接口。
- `ACTBasePolicy`：包装 `ACTBaseActionProvider`，实现 `BasePolicyProvider`。
- `SACBasePolicy`：直接从 rl-garden checkpoint 加载冻结 SAC policy 用于 residual base。
- `ZeroBasePolicy`：始终返回全 0 raw env action，用于 debug 训练。
- `make_base_policy()`：工厂函数，根据 `base_policy` 参数（`"act"` / `"sac"` / `"zero"`）构造对应 provider。

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

### Base policy 接口（`BasePolicyProvider`）

所有 base policy 实现 `BasePolicyProvider`（`nn.Module` 子类），对外只暴露三个方法：

- `select_action(obs) → BasePolicyOutput`：返回 raw env-space action。
- `reset(env_ids=None)`：在 env reset 时清空 provider 内部状态（如 ACT 的历史队列）。
- `bind_env(env)`：eval 时切换到 eval env（ACT provider 需要重建 state 来源）。

所有 provider 通过 `make_base_policy(base_policy=..., ...)` 工厂函数构造。

### Zero base（debug 模式）

RGBD 和 state 两个 peg launcher 都支持 `--debug`：

```bash
scripts/train_residual_sac_rgbd_peg.sh --debug
```

debug 模式等价于 `--base_policy zero`，始终返回全 0 raw env action，不加载任何 checkpoint。

### ACT base provider

非 debug 模式默认使用 ACT：

```bash
scripts/train_residual_sac_rgbd_peg.sh \
  --base_policy act \
  --base_ckpt_path act-peg-only
```

`--base_ckpt_path` 可以传完整 `.pt` 路径，也可以传名字。名字按
`pretrained_models/<name>.pt` 解析，默认 `act-peg-only` 对应
`pretrained_models/act-peg-only.pt`。也可以用 `$RL_GARDEN_PRETRAINED_DIR`
覆盖 pretrained 根目录。

`act-peg-only.pt` 是 state-only ACT checkpoint，期望 43 维 state 和 6 维
env action。peg RGB residual 训练的 actor/critic 仍然用 RGB dict observation；
ACT base provider 会在需要 base action 时从 peg env 的 `base_env` 重建这 43
维 full state，避免把 residual actor 的观测格式改成 state-only。

ACT provider 的输出是 raw env action，后续由 ResidualSAC 的 `ActionScaler`
转换成 normalized base action。

### SAC checkpoint 作为 base policy

可以把已有的 rl-garden SAC checkpoint 冻结后用作 base policy：

```bash
scripts/train_residual_sac_rgbd_peg.sh \
  --base_policy sac \
  --base_ckpt_path runs/<run_name>/checkpoints/final.pt \
  --base_sac_encoder resnet10 \
  --base_sac_image_fusion_mode per_key
```

相关参数：

- `--base_sac_encoder`：重建 base SAC 视觉编码器所用的 encoder 类型（`plain_conv` / `resnet10` / `resnet18`），须与保存 checkpoint 时的编码器一致。
- `--base_sac_encoder_features_dim`：编码器输出维度，默认 256。
- `--base_sac_image_fusion_mode`：图像融合模式（`stack_channels` / `per_key`），须与原训练保持一致。
- `--base_sac_deterministic`：默认 `True`，base policy 使用确定性预测。

`SACBasePolicy` 加载时会用 `validate_checkpoint_metadata` 校验 observation/action space 是否匹配；`strict=False` 可跳过严格校验。

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
  --base_policy act \
  --base_ckpt_path act-peg-only \
  --log_type tensorboard
```

其中：

- `--debug`：使用全 0 base policy，等价于 `--base_policy zero`。
- `--base_policy act`：使用 ACT 作为 base policy；`--debug` 会覆盖成 zero。
- `--base_policy sac`：使用已有 SAC checkpoint 作为 base policy（需配合 `--base_ckpt_path`）。
- `--base_ckpt_path act-peg-only`：加载 `pretrained_models/act-peg-only.pt`。
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
  --base_policy act \
  --base_ckpt_path act-peg-only \
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

## Peg Env 相机配置

`PegInsertionSidePegOnly-v1` 通过 `_default_sensor_configs` property 将
`base_camera` 默认分辨率定为 **128×128**。

所有 launcher（`train_sac_rgbd_peg.sh`、`train_residual_sac_rgbd_peg.sh`、
`eval_residual_sac_rgbd_peg.sh`）均默认传 `--camera_width 64 --camera_height 64`，
`make_maniskill_env` 会以此覆盖 `sensor_configs`，实际训练分辨率为 64×64。

若直接构造 peg env 而不指定 `camera_width`/`camera_height`，`base_camera`
分辨率为 128×128。加载已有 64×64 checkpoint 时需注意显式传入正确分辨率。

## Eval 脚本

`examples/eval_residual_sac_rgbd_peg.py` / `scripts/eval_residual_sac_rgbd_peg.sh`
在 `PegInsertionSidePegOnly-v1` 上评估保存的 RGBD peg ResidualSAC checkpoint，
并录制 `[base_camera | hand_camera]` 并排视频。

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_residual_sac_rgbd_peg.sh \
  --checkpoint_path runs/<run_name>/checkpoints/final.pt \
  --base_policy act \
  --base_ckpt_path act-peg-only \
  --num_eval_envs 16 \
  --num_eval_steps 100 \
  --output_dir runs/<run_name>/eval_residual_videos
```

脚本会从 checkpoint 元数据自动推断 encoder 类型和 `residual_action_scale`；
base policy 参数需手动与训练时保持一致。视频写入使用 ffmpeg（优先 `libx264`），
不需要额外安装 Python 视频库。
