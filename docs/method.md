# Sparse-WAM 方法与实现

本文说明当前 Sparse-WAM 的选择、稀疏执行和缓存语义。
full forward 仅为 step-0 conditional，随后同一步 unconditional 就使用稀疏选区。

| 方法机制 | 实现 |
|---|---|
| 对齐动作与未来帧，权重 (1,2,2,1)/6 | `scoring.profile_from_main_lse` |
| attention mass × (1 − normalized entropy)，Top6 | `selection.select_tokens` |
| 共同 Core 候选池与逐帧 Core80 | 同上，pool 大小 `spatial - 104` |
| Core union 外共享 Stable104 | 同上，分数 `mean/(1+CV)` |
| 主 attention 全可见 key 的 LSE，避免第二次 attention 输出 | Edge 主 LSE ABI；Nano `nano_layer` 单次 attention |
| 选中 token 真正经过 Attention/MLP，原始位置保留 | `controller` + `packing` |
| 当前选区 prediction + step0 缓存 prediction | `VelocityCacheSampler`，CFG 后、UniPC 前 |
| 每 chunk 刷新，chunk 内复用 | 每请求新建 `SparseWAMController` 和 sampler wrapper |

## 四步执行表

| step | conditional | unconditional | 视觉背景 |
|---|---|---|---|
| 0 | 全量，所有层打分，生成选区 | 稀疏，立即复用选区 | 用 full conditional 初始化，并缓存 CFG 后 velocity |
| 1 | 稀疏 | 稀疏 | 最近 full conditional 对应的缓存 |
| 2 | 稀疏 | 稀疏 | 同上 |
| 3 | 稀疏 | 稀疏 | 同上 |

只有第一行 conditional 打分，1 dense + 7 sparse。不能改成第一行两个分支都 dense。

## conditional-only full step 与 CFG 的衔接

conditional-only full forward 需要显式定义未选 unconditional 背景的初始化方式。
本实现沿用来源 Nano 分支 `piper_cosmos/cosmos3/k112_sparse.py` 的背景 bootstrap 处理，
将这一处理与新的 C80/S104 selector 组合，**不采用 K112 的选区策略**：

1. 记录 step-0 full conditional 的完整 `preds_vision`。
2. step-0 unconditional 稀疏 stack 恢复完整形状并经原生输出头后，在未来未选位置令 `u = c_full`。
3. 保持原生 CFG `u + g(c-u)`，所以该区域的初始 CFG delta 为零；L0、选区和 action 不做此替换。
4. 在 sampler wrapper 中缓存 step-0 的 CFG 后 vision velocity。
5. 后续每步在 `velocity_fn` 返回 CFG 后结果时，只将 future background 替换为缓存；action 和 L0 保持当前结果。
6. 原生 UniPC 接收合成后的完整 velocity，积分公式与 timestep schedule 不变。

该 bootstrap 是当前实现对 CFG 的明确近似，不是原生 dense CFG 的等价变换。
它避免用跳过 Transformer 的 unconditional 背景 hidden 初始化不可靠的 CFG 差值。

## hidden restoration 与 velocity cache 的区别

完整 hidden 恢复仍然保留：原生输出头需要原来的完整形状。
但未选位置的输出头结果不会成为后续未来背景的更新依据；该依据由 step0 的 velocity 缓存替换。
因此这不是仅重命名来源分支的 `current_stack_input` 恢复路径，而是真实接入了采样层缓存。

## 运行约束

- 当前支持单 rank、单样本、串行 conditional/unconditional CFG，固定 4 steps / CFG3 / shift5。
- 不支持 NATTEN、CP/CFGP/FSDP 多 rank、guidance interval、distilled FixedStepSampler、额外 control-CFG 或 prompt upsampling。
- 源码中的 token geometry 必须由真实 packed metadata 验证；不把 340/360 硬套到不匹配的 checkpoint。
- 主 LSE 必须覆盖真正可见的 UND + GEN keys，不能只在 future keys 中归一化。
- Sparse token IDs 与 RoPE 同步裁剪，不重新编号。
- Stable 是共享位置，不是跨帧共享 hidden feature。

## 证据边界

CPU 测试覆盖数学、生命周期和原生 packing 合同；Edge/Nano 已有真实模型的单 chunk GPU 验证。
具体输入、权重、计时范围和已验证数值检查见 [验证记录](validation.md)。
当前记录不建立完整闭环成功率，也不证明 compiled Sparse 与 eager Sparse 的普遍数值等价。
