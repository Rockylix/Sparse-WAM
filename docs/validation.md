# 本地验证记录

日期：2026-10-01。环境：Windows、Python 3.12.7、PyTorch 2.5.1+cpu，CUDA 不可用。

本表记录首次本地整理时的检查。后续按用户指定，将该目录初始化为独立 Git 仓库并以 `Rockylix/Sparse-WAM` 为发布目标；不继承来源仓库历史。

| 检查 | 实际结果 | 证明范围 |
|---|---|---|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q` | 31 passed，3.85 秒 | 数学、CPU adapter 合同及请求生命周期 |
| `python -m ruff check src tests tools` | 通过 | 新核心、测试与整理工具的静态 lint |
| `python -m ruff format --check src tests tools --exclude layout.py` | 23 files already formatted | `layout.py` 保留来源文件原字节，不做格式重写 |
| Python AST 解析 | 1230 个文件通过 | 包括 vendor；只证明语法可解析 |
| `python tools/check_sources.py` | 1357 个来源文件、1 处登记修改全部匹配 | 固定快照与本地 Edge scorer 接线的来源完整性 |
| Edge / Nano `config` CLI | 均成功 | 无需加载权重即可导出固定配置 |
| `serve --help` | 成功 | CLI 参数解析；不代表服务已启动 |
| 两份来源 checkout 的 Git status | 均无改动 | 原分支源码保持只读 |
| 新目录检查 | 无 `.git`、symlink、`.pt` / `.pth` / `.safetensors` | 没有复制 Git 历史或模型权重；没有创建/上传远端 |

## 关键测试覆盖

- 与来源 Edge selector 的参考实现比较选区；验证 Core80/Stable104 不相交、共享 Stable、逐帧 Core 和零 attention 的有限退化行为。
- 主 LSE scorer 与全可见 key softmax 数值比较，覆盖 GQA 和不同 UND 长度。
- 对 Nano layer 使用小型张量/reference attention，验证 profiling 只调用一次主 attention，并保持输出一致。
- 通过实际 controller/hooks、packing adapter 和 sampler wrapper 验证 28/36 层的完整八次 CFG forward；网络权重用可审计测试替身。
- 验证 step-0 conditional 背景 bootstrap、CFG 后且 UniPC 前的 velocity 替换，以及 L0/action/ROI 保持当前结果。
- 连续两次 PaperRunner 请求使用不同输入，确认背景 cache 更新、不跨 chunk 复用；失败后清除 hooks，不把失败报告为成功。
- 验证真实原生 packing 函数的 metadata、RoPE、异长 CFG 和恢复。为隔离缺失的 `loguru`，测试执行源码 AST 时只移除 logger import；不宣称完整 framework 已成功导入。
- 验证结构化 caption 保留任务文本，以及 Dense 对照准确记录 8 dense / 0 sparse、无 scorer/cache；拒绝不兼容采样参数。

## 首次整理时尚未验证（2026-10-01）

没有加载 Edge/Nano 权重，没有完成完整 Cosmos 依赖安装或 GPU 服务启动。
真实 attention kernel 的 LSE、真实 checkpoint/预处理的 token 布局、Nano offload、编译/CUDA Graph、显存、延迟和 RoboLab 成功率都仍需目标环境验证。
本机缺少 `loguru` 等完整 runtime 依赖；依赖补装尝试没有成功，未用未运行的原仓库集成测试充当通过证据。

本次接入 velocity cache 与 Nano 主 LSE 是行为变化；这里只能报告实现和局部验证，不能宣称论文效果已复现。
下一阶段具体检查见 [Edge/Nano 差异与 GPU 验收](backbone_differences.md)。

## 4090 配对冒烟（2026-10-06）

直接使用本仓库 `src/sparse_wam` 和各自的 `vendor/edge`、`vendor/nano`，
没有导入旧实验分支的 ASI controller。远端起点为 `92a770891c95e3a0ab15ffe4866e9a2a2e7c0961`。
本轮修改 Nano 动态网格校验和报告；测速工具新增 Dense eager/编译 Sparse 隔离与 replay 计数。
未修改 Core80/Stable104、Top6、评分公式、稀疏调度或 velocity-cache 计算。

环境：RTX 4090 24GB，driver 580.178.04，PyTorch 2.10.0+cu130。
Edge 使用项目内既有 `.venv`；Nano 经用户批准使用既有宿主机 Cosmos `.venv`，
强制 reasoner offload / instruction KV cache，GEN 常驻；未连接机器人或仿真器。
本机没有 `cosmos3-dev` 容器，Nano 结果只代表这次获批环境。

两种骨干均使用本地官方 DROID checkpoint、同一 BananaInBowl observation。
输入由已有录制 batch 的首个 condition 图像/状态重建，图像已经预处理过，
不是原始相机输入的逐位重放。输入来源见 `outputs/smoke4090_20261006_v1/input_provenance.json`。
4 steps、guidance=3、shift=5，3 轮成对预热、5 轮成对测量；
每轮 Dense/Sparse 使用相同输入和 seed，执行先后交替。

| 模型 / 执行路径 | Mean (s) | Median (s) | P90 (s) | 相对本模型 Dense median |
|---|---:|---:|---:|---:|
| Edge Dense eager | 0.891369 | 0.889850 | 0.894290 | 1.000× |
| Edge Sparse compile+Graph | 0.498788 | 0.498949 | 0.499136 | 1.783× |
| Nano Dense eager | 3.192250 | 3.196744 | 3.199948 | 1.000× |
| Nano Sparse compile+Graph | 1.832512 | 1.832207 | 1.840102 | 1.745× |

计时是 CUDA 同步的 `service.infer`：输入预处理、generation、动作后处理和检查；
不包含模型加载、首次编译和首轮指令 prefill；不解码 future video。
这是稀疏与编译的联合收益，不能单独归因于 token sparsity。
测量轮 Dense 均为 0 次 Graph replay；Edge Sparse 每 chunk 264 次，Nano Sparse 288 次。
Graph 是 layer/函数级的编译器托管执行，不是一个全局四步采样 Graph。
Edge 编译 decoder 与 encode/decode heads；Nano 使用本仓库已有 GEN adapter 的编译路径。

本输入 Edge/Nano 都是 17×20=340 spatial tokens/帧，每帧保留 K184；
完整/稀疏 GEN 长度为 3093/1845，L0 和 action 始终保留。
Nano 不再硬编码 360，CPU 测试覆盖 340/360/400 网格和预算过小拒绝。
所有测量完成，action/vision 通过有限值检查；一次全量、七次稀疏、七次 pack/restore 和四次 velocity update 均通过运行时检查。
Nano 日志记录 cond/uncond 各一次 UND prefill，随后卸载 UND/bridge；本次进程峰值 allocated/reserved 为 17.245/17.773 GiB，没有 OOM。

原始样本、action MSE 和运行参数分别保存在：

- `outputs/smoke4090_20261006_v1/edge_compile_graph.json` 及同名 `.log`
- `outputs/smoke4090_20261006_v1/nano_compile_graph.json` 及同名 `.log`

复现：在本仓库，沿用原有 observation/native JSON 和本地 checkpoint，
运行 `tools/benchmark_policy.py --backbone edge` 或 `nano`，
加上 `--execution compile-graph --dense-eager --warmup 3 --repeats 5 --seed 0`。
设置 `PYTHONPATH=src`、`LD_LIBRARY_PATH=`、`HF_HOME=/root/cosmos3/cosmos/checkpoints/hf_home`，
以及 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 OMP_NUM_THREADS=4`。
Nano 另设 `COSMOS3_REASONER_OFFLOAD=1 COSMOS3_REASONER_KVCACHE=1 COSMOS3_POLICY_REASONER_CACHE=1 COSMOS3_LAYER_OFFLOAD=0`。

36 项 CPU 测试、ruff check、1357 份来源文件校验通过。
本轮没有验证闭环成功率、不同图像/指令 cache 失效、多尺寸 GPU 推理，
也没有证明 compiled Sparse 与 eager Sparse 数值等价；这些不能由本次冒烟推导。

### 同日计时边界核查

额外同步诊断（3 轮预热、5 轮测量，不修改模型算术）将 `infer` 与原生 generation 分开：

| 模型 | Dense generation median | Sparse generation median | generation 加速 | 完整 infer 加速 |
|---|---:|---:|---:|---:|
| Edge | 0.852108 s | 0.461482 s | 1.846× | 1.782× |
| Nano | 3.152333 s | 1.786813 s | 1.764× | 1.747× |

服务外围增加约 38–40 ms，不能将完整 infer 比值与历史 generation 比值直接比较。
Edge 的 1.846× 与原项目历史 B6 表的 1.841× 接近，主 LSE、批量评分、
首轮 decoder 编译、索引/metadata 复用和 Graph 均已保留。
本输入每 chunk 原始 packing 调用数：Edge 3 次、Nano 9 次（均包含输入准备）。
Nano 尚未接入真机服务的请求内原始 packing 复用；旧 Nano 测速脚本也未接入，
因此不能直接把新旧速度差归因于该优化。历史 Nano 是 Piper 20k/HDF 的 360-token 网格，
本次是官方 DROID 的 340-token 网格；历史普通 ASI 与本仓库 velocity-cache 版本也不是同一策略。
诊断原始数据在 `outputs/smoke4090_20261006_v1/boundary_comparison.json`。

## Nano 请求内打包复用回归（2026-10-06）

先将上述 Edge 完整验证版推送至 `origin/main` 的 `15c7e69`，再单独实现此优化。
仅 Nano Sparse 默认启用；Dense 保持原生 eager，Edge 继续使用原生模板复用。
可用 `--no-nano-packing-cache` 关闭，不改变评分、Core80/Stable104、velocity cache 或 UniPC。
复用只覆盖原生 `_get_velocity` 内的 packing，输入准备仍重新构造。
更新当前 noisy tokens 的原生代码仍执行，缓存命中时只刷新 noisy-position timestep；
RoPE、condition mask 和 CFG 文本布局不改变。缓存按请求销毁，异常时恢复模型方法。

与上面的 Nano 官方 DROID checkpoint、环境、同一 observation 来源保持一致。
本次同模型交替 Dense eager、Sparse compile+Graph packing off/on：3 轮预热、10 轮测量。
轮次交替使用 seed 0/1；seed 1 时图像所有通道增加 1（截断至 uint8 范围），
每轮三个模式仍使用完全相同输入。这样覆盖两组输入，而非仅重复同一次输出。
generation 与 infer 分别同步计时，后者含前后处理；不解码视频、不连接机器人。

| 路径 | generation Mean | Median | P90 | 相对 Dense median |
|---|---:|---:|---:|---:|
| Dense eager | 3.170534 s | 3.171263 s | 3.185095 s | 1.000× |
| Sparse compile+Graph，packing off | 1.802849 s | 1.803311 s | 1.814061 s | 1.759× |
| Sparse compile+Graph，packing on | 1.798973 s | 1.800472 s | 1.811374 s | 1.761× |

两个 median 相差 2.839 ms；按每轮 off−on 配对计算，平均节省 3.876 ms、中位节省 4.739 ms。
总收益很小，不能将此优化写成明显的整 chunk 加速。
完整 infer median 分别为 Dense 3.212422 s、off 1.844703 s、on 1.841892 s。

- 13 组成对回归全部通过逐位相等：原始 action、vision latent、执行 mask、Top6 层、
  block quality 及服务返回 action。结论仅限这两组输入，不是所有输入等价的证明。
- 每 chunk packing 调用仍为 9 次，其中 3 次构建、6 次缓存命中；3 次包括输入准备及两支 CFG。
- Sparse off/on 每 chunk 均为 288 次 Graph replay，Dense 为 0；全进程 UND prefill 仅 2 次。
- 51 项 CPU 测试通过，包括禁用路径、请求清理、文本/布局隔离、异常恢复与不支持布局回退。
- `ruff check`、`git diff --check`、1357 份 vendor 来源校验通过。

工具：`tools/benchmark_nano_packing.py`；原始结果：
`outputs/smoke4090_20261006_v1/nano_packing_on_off.json`（同名 `.log`）。
调用参数同前述 observation/native-config/checkpoint，增加 `--warmup 3 --repeats 10 --output <new.json>`。
工具本身固定 Nano Sparse compile+Graph、Dense eager，不继承其他策略开关。

## Nano Piper HDF5 配对测速（2026-10-06）

使用当前本地 Sparse-WAM（含 Nano packing cache），不调用 Piper 的旧 ASI controller。
只新增 `tools/benchmark_nano_hdf5.py` 作为数据/测量入口；算法与 vendor 源码未修改。
模型来自本仓库 `vendor/nano/cosmos_framework`，HDF reader 和 batch collator 来自 Piper。
本轮继续使用此前用户批准的宿主机 Nano 环境；不代表真机部署容器性能。

- 权重：`/root/piper-cosmos3-quant/cosmos_battery/20k`。
- 数据：`/root/piper-cosmos3/hdf_data/perfect/episode_10.hdf5`，dataset sample 64 / frame 64。
- 三视角 concat_view，Piper14 state/action，指令 `Assemble the mouse's battery.`；caption 显式 JSON 序列化。
- 相同完整 batch、seed=0、guidance=3、4 steps、shift=5；输入副本在主计时区外建立并校验哈希。
- 3 轮预热、20 轮交替 Dense/Sparse；Dense eager，Sparse compile+CUDA Graph。
- 主口径：CUDA 同步的 `generate_samples_from_batch`，包括本次生成过程中的评分、稀疏与 velocity cache；
  不含 HDF 预处理、输入复制、初次 prefill/编译、RPC、VAE decode。另记含 controller 设置与输出检查的 runner 时间。

| 路径 | Mean | Median | P90 | Median speedup |
|---|---:|---:|---:|---:|
| Dense eager | 3.335734 s | 3.340921 s | 3.353251 s | 1.000× |
| Sparse-WAM compile+Graph | 1.834787 s | 1.838814 s | 1.846538 s | 1.817× |

包含 runner 开销的 median 分别为 3.341111 / 1.839250 s。
网格按实际输入识别为 20×18=360，而不是硬编码；每 future frame 保留 Core80+Stable104=184（51.11%）。
GEN 序列从 3273 缩为 1865，保护 L0 和全部 action token；每 chunk 一次 full profile、七次 sparse forward。
Top-6 为 `[23,22,21,24,19,25]`。Sparse 每个测量 chunk 均有 288 次 Graph replay，Dense 为 0。
原始 packing 每 chunk 9 次调用、3 次构建、6 次命中；Reasoner 全进程只 prefill 两次，结束时 UND decoder CUDA 权重为 0，GEN 权重 12.94 GiB。
峰值 allocated/reserved 为 17.266 / 17.793 GiB。

所有 action/vision 输出通过 NaN/Inf 检查；相同模式的固定输入重复结果逐位一致，原始 batch 哈希未变化。
Sparse 对本轮 Dense 的完整返回 tensor 误差（不是物理关节角误差）：

| 输出 | MSE | Relative L2 | Cosine |
|---|---:|---:|---:|
| action | 0.00139243 | 0.0501787 | 0.998744 |
| vision latent | 0.0230565 | 0.234151 | 0.972232 |

历史相同 Piper checkpoint/sample 配置的普通 ASI 测得 3.352118/1.853345=1.809×；
当前结果与该历史加速比接近，但当前包含 step0 背景 velocity cache，不能据此声明两版本算法等价。
也不能将本次与官方 DROID/340-token 试验的差异归因于某一个优化。
没有连接机械臂、没有闭环成功率，也没有解码未来帧。

完整结果、逐轮时间、输入哈希、caption、source 状态与复现参数：
`outputs/nano_hdf5_piper20k_s64_compile_graph_v3/results.json`。
复现使用该文件的 `argv`，并更换 `--output-dir` 防止覆盖；环境变量与前述 Nano 测试一致。
`v1/v2` 为测试入口的 VAE 后缀与标量哈希错误尝试，未纳入任何有效样本。
现有 51 项 CPU 单元测试、标量/bfloat16 输入哈希检查、ruff、vendor source 校验均通过。
