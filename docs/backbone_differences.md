# Edge / Nano 的方法与执行差异

本表针对 `SOURCE_MANIFEST.json` 固定的方法分支与论文 policy 布局；其他 checkpoint 不保证具有相同几何。

| 项目 | Cosmos3 Edge | Cosmos3 Nano |
|---|---|---|
| 生成层数 / hidden width | 28 / 2048 | 36 / 4096 |
| MLP | 两层 Linear，ReLU²；intermediate 9216 | 三矩阵 SwiGLU；intermediate 12288 |
| 每帧空间 token | 340，17×20 | 360，20×18 |
| 完整 GEN 长度 | 3093 | 3273 |
| 稀疏 GEN 长度 | 1845 | 1865 |
| 未来视觉 token 保留率 | 184/340 ≈ 54.12% | 184/360 ≈ 51.11% |
| 主 attention LSE | 使用来源 B6 decoder 已有的返回接口 | GEN adapter 的主 attention 一次调用同时返回输出与 LSE |
| Transformer 执行 | 调用原生 Edge decoder | 提取来源 ASI 的 GEN QKV、RoPE、attention、MLP 路径 |
| 模型驻留 | 原生 Edge 服务路径 | Reasoner prefill/offload，GEN 常驻，原生指令 KV cache |
| compile / Graph | 原生 layer compile 设置 | 单独编译 GEN adapter；Dense 对照保持原生 eager |

两者共享 Core80/Stable104、Top6、动作时间对齐、层质量、Stable CV、主 LSE 评分定义、CFG 调度与背景 velocity cache。
完整 GEN 长度为 `9×spatial+33`；稀疏长度为 `spatial+8×184+33`。
33 个 action 位置包含一个历史/条件位置及 32 个预测位置。L0 全量保留，文本 UND 不计入上述 GEN 长度。

同一个 K184 对 Nano 的未来视觉剪枝比例更高，但不能直接据此推导速度收益。
Nano 的矩阵规模、MLP、权重驻留与指令 cache 行为都会影响延迟；Edge 的评分和 Python 调度固定开销占比可能更大。
这是依据执行结构的判断，尚无本次 GPU profiler 测量支持具体比例。

两个 runtime 使用同名 `cosmos_framework`，目前保持固定快照并在独立进程/环境加载。
在没有真实模型输出回归前，不把 138 个内容不同的同路径源文件直接覆盖成一个版本。
原生指令 KV 的跨 chunk 复用与 Sparse-WAM 的 chunk 内 mask/velocity 复用具有不同生命周期。

## 相对来源方法分支的变化

- 两个指定方法分支原来都主要使用未选 hidden 的 `current_stack_input` 恢复；整理版新增采样层 velocity cache。
- Edge 的主 LSE 路径原来是 B6 可选路径；整理版入口固定使用它。
- Nano ASI 原来额外计算 action-row attention 的 LSE；整理版改为复用主 attention LSE，并补上统一 RoboLab 服务入口。
- Dense/Sparse 都先序列化结构化任务文本。Nano 旧 HDF 测速曾记录任务文本未注入，其旧数字不能用来验证当前语言条件路径。

因此本次包含方法实现补齐，不是保持历史输出不变的纯目录搬迁。旧测速与成功率不自动继承。

## GPU 验收顺序

1. 分别在 Edge/Nano 对应环境加载真实 policy checkpoint，确认预处理后的布局和完整 Dense 输出。
2. 运行 eager Sparse，核对有限输出、1 dense/7 sparse、七次 pack/restore、每帧 184，以及真实主 attention LSE。
3. 连续运行不同 observation 和不同 instruction，检查每 chunk 新选区、背景 cache 刷新及原生指令 KV 失效。
4. 使用相同输入/seed 的 Dense/Sparse 配对测速，区分首请求、指令 cache miss、稳态 cache hit。
5. 再验证 compile / compile-graph 的数值与性能；单独标明 Nano Dense eager / Sparse compiled 的对照。
6. 使用论文完整 RoboLab 协议重新测量闭环成功率。离线 action MSE 不替代任务成功率。

上述 GPU 步骤当前均未完成，参见 [本地验证记录](validation.md)。
