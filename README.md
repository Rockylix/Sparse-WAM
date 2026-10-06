# Sparse-WAM for Cosmos3

面向 **Cosmos3-Edge / Cosmos3-Nano policy** 的 Sparse-WAM 整理版。
核心包含 Action-Guided Token Selection、Pilot 主 attention LSE 评分、固定选区稀疏执行、背景 velocity cache。

来源限定为用户指定的两个方法分支，不使用它们的 main：

| 后端 | 来源分支 | 固定提交 |
|---|---|---|
| Edge | `version/core80-stable104-action-weighted` | `deb4af65000d06b7b13e30c6ebcc0d44e6c67921` |
| Nano | `version/asi` | `a89509e1f54e60f0622741995efd5e2f3d6710d6` |

**当前状态：代码已整理并通过 CPU 单元/数值检查；没有在本机加载权重、启动 GPU 服务或复测论文结果。**
相对于来源分支，接入 velocity cache 和 Nano 主 LSE 复用是实质行为变更，不能沿用旧测速或成功率。

## 唯一主线配置

- Core80 + Stable104，每帧 K184；Top6 informative layers。
- 四个对齐 action queries 的权重为 `(1,2,2,1)/6`；Stable 分数 `mean/(1+CV)`。
- 4 denoising steps，CFG=3，shift=5；L0、文本、全部 action 保留。
- **Full step 只指 step 0 的 conditional 分支**，不是两个 CFG 分支都 dense。
- step 0 unconditional 立即使用 conditional 选区；全程 **1 dense + 7 sparse forwards**。
- 只在每个 chunk 内复用选区/packing metadata；每个新 chunk 创建新控制器和 velocity cache。
- 缓存应用在 native CFG 之后、UniPC 积分之前；L0 与动作保持当前步输出。

完整定义见 [论文对齐说明](docs/paper_alignment.md)，尤其是 step-0 背景 CFG 的初始化规则。
两种模型的执行差异见 [Edge/Nano 对比](docs/backbone_differences.md)；已执行的检查与未验证部分见 [验证记录](docs/validation.md)。

## 目录

```text
src/sparse_wam/
  config.py        唯一参数定义与不兼容配置检查
  selection.py     论文 Core / Stable 选择
  scoring.py       Pilot：主 attention LSE + 对齐局部 QK
  controller.py    一次 dense / 七次 sparse 生命周期
  packing.py       原位置 gather/scatter、CFG 异长 metadata 缓存
  velocity.py      公式（7）及 native UniPC wrapper
  backends/        Edge 原生 decoder 与 Nano GEN-resident 执行
  runtime.py       独立 backend 选择与 PaperRunner API
  service.py       统一 RoboLab/openpi 服务接线
configs/           从 config.py 导出的可核对配置记录
tests/             CPU 数值、调度、缓存、原生 packing 合同测试
tools/             配对 benchmark 与来源校验
vendor/edge/       Edge 固定 Cosmos runtime 快照
vendor/nano/       Nano 固定 Cosmos runtime 快照
```

两个 runtime 版本有差异，目前隔离加载；**一个进程只加载一个后端**。
vendor 保留框架的内部依赖与配置注册，仍含其训练/多模态基础设施；它们不构成本项目的公开方法入口。
本阶段没有假装完成全部 runtime 的最小化裁剪。核心方法中不包含 K112、Core64、ToCa、WorldCache 或跨 chunk residual cache。
底层 `_k112_sparse_controller` 是 Nano 来源框架的 hook 名称，不表示运行 K112 策略。

## 本地核心检查（不需要模型权重）

在项目根目录、已有 PyTorch/pytest 环境中：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
PYTHONPATH=src python -m sparse_wam config --backbone edge
PYTHONPATH=src python -m sparse_wam config --backbone nano
python tools/check_sources.py
```

PowerShell 使用 `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'` 和 `$env:PYTHONPATH='src'`，然后执行相同 Python 命令。

## 目标 GPU 环境

完整推理需要 Linux、CUDA、checkpoint、对应 VAE/tokenizer 及 Cosmos 依赖。
建议分别使用两个来源分支已经验证的环境，不直接升级 runtime。
若需重建环境，可在独立环境中按各自 vendor 的 `pyproject.toml` / `uv.lock` 选择 CUDA group。
例如来源使用的完整 CUDA 13 环境：

```bash
uv sync --project vendor/edge --all-extras --group cu130-train --group policy-server
uv pip install --python vendor/edge/.venv/bin/python --no-deps -e .
# Nano 在另一个环境：把上述 vendor/edge 替换为 vendor/nano。
```

这保留来源框架的较宽依赖集合，并非已验证的最小安装集。首次安装可能需要网络/平台特定 wheel。
本项目采用完整 source checkout 的 editable 安装；不将独立 wheel 当作包含 vendor 的完整部署包。
模型权重不放入仓库；使用本地 checkpoint 及其原始 config。不要用 Piper14 权重冒充 DROID。

## RoboLab 服务

```bash
PYTHONPATH=src vendor/edge/.venv/bin/python -m sparse_wam serve \
  --backbone edge --checkpoint-path /path/to/Cosmos3-Edge-Policy-DROID \
  --mode sparse --execution eager --host 127.0.0.1 --port 8000 --seed 0

PYTHONPATH=src vendor/nano/.venv/bin/python -m sparse_wam serve \
  --backbone nano --checkpoint-path /path/to/Cosmos3-Nano-Policy-DROID \
  --mode sparse --execution eager --host 127.0.0.1 --port 8000 --seed 0
```

`--mode dense` 是原生全量对照；两种模式均执行相同的任务文本序列化。
原生 checkpoint/domain/VAE 配置可通过 `--native-config /path/to/native.json` 传入。
该 JSON 使用对应版本 `RobolabServerArgs` 的字段，不覆盖固定论文参数。
Edge 保留 340 个空间 token 的布局检查；Nano 从真实 packed layout 读取空间尺寸，
不把某次输入的 360 tokens 写死为模型属性。官方 DROID 默认输入为 17×20=340；
其他预处理可产生不同网格。K184、L0/action 保留、原始位置编码及同 chunk 布局一致性检查不变。

Nano 自动启用原生 Reasoner offload / instruction KV cache；禁止与逐层全模型 offload 混用。
指令 KV 可由原生框架跨 chunk 缓存，Sparse-WAM 的 mask/背景 velocity 不跨 chunk。
Nano Sparse 默认复用同 chunk 的原始序列打包 metadata；两支 CFG 按文本/布局独立缓存。
只在原生 `_get_velocity` 内复用，noisy tokens 仍由原生调用方每次更新，timestep 原地刷新。
输入准备不使用该缓存，多样本/逐帧 timestep/sound/多视觉 item 等情况回退原路径。
可用 `--no-nano-packing-cache` 关闭作消融；Dense 不启用该优化，Edge 保留原生缓存路径。
保留原生服务的图像处理、state/action 归一化、历史行裁剪、gripper 翻转及推进 seed 的语义。

`--execution compile` / `compile-graph` 是可选执行后端。Graph 是编译器管理的 layer 路径，
不是整个四步采样被一个 CUDA Graph 包裹。编译可能改变浮点归约、Top-K 边界及动作输出。

## 配对 benchmark

`tools/benchmark_policy.py` 接收一个录制的 openpi observation `.npz`，不连接机械臂。
NPZ 的键与 RoboLab observation 相同；图片/状态为 NumPy 数组，prompt 为 Unicode 标量，不允许 pickle。

```bash
PYTHONPATH=src vendor/edge/.venv/bin/python tools/benchmark_policy.py \
  --backbone edge --checkpoint-path /path/to/checkpoint \
  --observation /path/to/observation.npz --execution eager \
  --warmup 5 --repeats 30 --seed 0 --output outputs/edge_eager.json
```

脚本在同模型、同 observation、成对 seed 下交替 Dense/Sparse，并报告 median/P90、原始样本和 action MSE。
比较 Dense eager 与 Sparse compile + Graph 时，使用 `--execution compile-graph --dense-eager`。
该选项让 Edge 的 Dense 调用原生未编译 layer/head，Sparse 调用原生编译包装器；不替换模型算法。
结果中的 `cuda_graph_replays_by_round` 记录真实 replay 调用次数，而非仅记录配置开关。
`tools/benchmark_nano_packing.py` 在同模型中交替 Dense eager、Sparse packing off/on，
同时报告 generation/infer 时间并检查 action、vision、mask、选中层和 quality 的逐位一致性。
它使用两组 observation/seed 变体做有限回归，不代表所有输入上的数值等价或闭环成功率。
**计时包含 `policy.infer` 的预处理、推理和动作后处理**，不同于来源文档的 generation-only 时间。
Nano Dense 保持原生 eager，即使 sparse adapter 开启 compile；必须按输出记录区分算法收益与编译收益。
实际 RoboLab success rate 需要另行运行完整闭环协议，不能由这个离线脚本或 action MSE 推断。

## 来源与仓库边界

本项目使用全新的独立 Git 历史，没有复制来源仓库的 `.git`、历史分支或权重。
发布目标为私有仓库 `Rockylix/Sparse-WAM`；只配置该仓库为 `origin`，不配置 NVIDIA `upstream`。
来源记录在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，本地变更见 [修改说明](docs/changes.md)。
第三方源码的许可证、SPDX 和适用的版权/来源声明保留，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
