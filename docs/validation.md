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

## 尚未验证

没有加载 Edge/Nano 权重，没有完成完整 Cosmos 依赖安装或 GPU 服务启动。
真实 attention kernel 的 LSE、真实 checkpoint/预处理的 token 布局、Nano offload、编译/CUDA Graph、显存、延迟和 RoboLab 成功率都仍需目标环境验证。
本机缺少 `loguru` 等完整 runtime 依赖；依赖补装尝试没有成功，未用未运行的原仓库集成测试充当通过证据。

本次接入 velocity cache 与 Nano 主 LSE 是行为变化；这里只能报告实现和局部验证，不能宣称论文效果已复现。
下一阶段具体检查见 [Edge/Nano 差异与 GPU 验收](backbone_differences.md)。
