# RNG 与数值非确定性注意事项

本文档记录两类本质不同、但容易被混为一谈的"非确定性"问题，都是在
RGB ResNet PickCube SAC 调试过程中踩坑后总结的经验：

1. **RNG 状态隔离**——诊断/日志代码意外消耗了训练 RNG 流，悄悄改变后续
   探索噪声，破坏可复现性。
2. **浮点 reduction-order 不可复现**——`critic_impl="vmap"` 与
   `critic_impl="legacy"` 在相同 seed/输入下从一开始就不是 bit-identical，
   这与 RNG 无关，而是 cuBLAS kernel 选择导致的浮点累加顺序差异。

两者的修复方式完全不同，混淆会导致排查方向跑偏。

---

## 第一节：RNG 状态隔离 —— actor diagnostics 的 `fork_rng`

### 问题本质

训练循环中为了记录诊断指标（例如 `policy.actor_diagnostics()` 里对
action 分布做一次额外的 `rsample()`），如果不做隔离，这次额外采样会
"偷走" RNG 流中的随机数，悄悄改变后续训练步骤中 actor 探索噪声、
dropout 等的取值。这个副作用极其隐蔽：训练仍然能跑、loss 曲线看起来
正常，但同一份代码在"开诊断"和"关诊断"两种配置下会产生不同的训练
轨迹，破坏可复现性和消融实验的可比性。

### 演进历史

这个问题在 main 上经过 4 个 commit 才收敛为现在的实现：

- `790ad9f feat(sac): log entropy/alpha diagnostics for alpha-collapse analysis`——
  最初引入诊断日志，直接在训练循环里调用
  `policy.actor_diagnostics(obs)`，未做任何 RNG 隔离。
- `ef7b58a hotfix: isolate actor diagnostics RNG`——发现诊断采样会扰动
  训练 RNG，用 `torch.random.fork_rng` 紧急修复。
- `f504a07 fix(residual): fix log entropy/alpha diagnostics for residual policy; open use-critic-layer-norm`——
  为 residual 策略的诊断（需要额外传入 `base_actions`）补齐相同的隔离。
- `fb14e5b fix(sac): preserve diagnostics rng for residual sac`——把隔离
  逻辑收敛为一个统一的 sealed wrapper，子类不再需要各自处理 `fork_rng`。

### 当前实现

`rl_garden/algorithms/sac_core.py` 中：

- `_actor_diagnostics(data)`：sealed wrapper，用
  `torch.random.fork_rng(devices=devices)` 包裹诊断计算（`devices`
  根据 `self.device` 是否为 CUDA 决定是否传入当前 CUDA device index）。
  调用前后 CPU/CUDA RNG 状态完全一致，无论诊断内部做了多少次
  `rsample()`。
- `_compute_actor_diagnostics(data)`：子类应该 override 的 hook，默认
  实现是 `self.policy.actor_diagnostics(data.obs)`。`ResidualSAC`
  （`rl_garden/algorithms/residual.py`）override 为
  `self.policy.actor_diagnostics(data.obs, data.base_actions)`，因为
  residual 策略的诊断需要额外的 base action 输入。

`tests/test_sac_core.py` 中的 `test_actor_diagnostics_preserves_cpu_rng_state`
和 `test_actor_diagnostics_preserves_cuda_rng_state` 是这个不变量的回归
测试：以固定 seed 分别在"调用诊断前"和"调用诊断后"各采样一次 `torch.rand`，
两次结果必须完全一致。

### 经验 / 不变量

任何"非训练关键路径"上调用随机采样的代码（日志、诊断、eval 探针等），
都必须包裹在 `fork_rng` 中。扩展诊断逻辑时：

- **只 override `_compute_actor_diagnostics`**，不要直接修改
  `_actor_diagnostics`——否则会丢失隔离保证，且容易在不同子类间重复
  实现 `fork_rng` 样板代码。
- 如果新增的诊断需要额外的 batch 字段（例如 residual 的
  `base_actions`），通过 override `_compute_actor_diagnostics` 从
  `data` 中读取，而不是改变 `_actor_diagnostics(data)` 的签名。

---

## 第二节：浮点 reduction-order 不可复现 —— vmap vs ModuleList critic ensemble

### 现象

`rl_garden/networks/actor_critic.py` 中的 `EnsembleQCritic` 有两种实现
（由 `critic_impl: Literal["vmap", "legacy"]` 选择）：

- `critic_impl="legacy"`：`nn.ModuleList`，N 个 Q-head 各自独立做一次
  `(batch, in) @ (in, hidden)` matmul → cuBLAS 单 GEMM kernel。
- `critic_impl="vmap"`（默认）：`torch.func.stack_module_state` 把 N 个
  Q-head 的参数沿 axis 0 堆叠，再用 `torch.func.vmap` 做一次 batched
  GEMM `(N, batch, in) @ (N, in, hidden)` → cuBLAS strided batched GEMM
  kernel。

| 场景 | 表现 |
|---|---|
| State-only SAC | 两者数值接近，训练结果等价 |
| RGB ResNet PickCube SAC（固定 batch 单步 parity test） | 数值等价 |
| RGB ResNet PickCube SAC，训练 8k-96k 步 | 两者 metrics 仍然 bit-identical |
| RGB ResNet PickCube SAC，136k 步之后 | 开始分歧，vmap 的 Q 值比 legacy 高
约 8-14%；某些 seed 下 vmap 训练失败（success 0% vs legacy 87.5%） |

### 根因：这不是 RNG 问题

两条路径用相同 seed、相同初始化、相同输入，**从训练一开始浮点层面就
不是 bit-identical**——但差异小到在 8k-96k 步内不影响 metrics 的可见
精度。根因是 cuBLAS 对 batched GEMM（vmap）与单次 GEMM（legacy）会
选择不同的 kernel，而不同 kernel 的浮点累加（reduction）顺序在
IEEE 754 下不满足结合律（"batch invariance" 问题，参见
Thinking Machines Lab 2025 / arXiv:2511.00025：相同输入下
`torch.mm(x, W)` 与 `torch.mm(batch, W)[0]` 的差异可超过 1600 ULP）。

这个 ULP 级别的初始差异在训练中逐步放大：critic 输出的微小差异 →
梯度方向微小偏移 → encoder 参数更新路径偏移 → rollout 轨迹偏离 →
replay buffer 采样到不同样本 → 后续更新进一步偏离 → 136k 步后出现
质变的训练结果差异（alpha 单调衰减、策略过早收敛、无法探索到稳定
抓取轨迹）。

### 为什么 state-only SAC 不受影响

MLP encoder 的优化景观低维且平滑，微小的 critic 数值差异在梯度下降中
会被自行修正；ResNet encoder 的高维参数空间对微小梯度差异更敏感，更
容易被推入不同的吸引盆，从而把 ULP 级别的浮点差异放大成训练结果上的
质变。

### 实践指引

- `critic_impl` 开关已存在于 `rl_garden/networks/actor_critic.py` 和
  `SACTrainingArgs`（CLI: `--critic_impl {vmap,legacy}`）。
- RGB ResNet 训练（如 PegInsertionSidePegOnly、PickCube RGB）建议显式
  设置 `--critic_impl legacy`，避免上述长程发散。
- state-only SAC 不受此问题影响，`vmap`（默认值）即可。
- vmap 仍然是一个合法、性能更好的实现，只是与 legacy 不 bit-identical；
  不要把它当作 legacy 的纯粹加速版本来做精确数值对比（例如 ablation
  对比实验应固定 `critic_impl`，不要在 vmap/legacy 之间切换）。

详细的数值分析、第三方框架（SB3 / RLPD / WSRL）实现对比、以及
短/中/长期建议（`alpha_min`、`target_entropy`、`q_lr`、
`torch.use_deterministic_algorithms`、`torch.compile` + `vmap` 等），
见 Obsidian 笔记《vmap vs ModuleList Critic Ensemble 数值分析》
（`30 Zettelkasten/32 Permanent/Reinforcement Learning/`，2026-06-02）。
本文档只总结要点，不重复其完整推导。
