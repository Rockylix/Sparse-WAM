# Historical local validation note

Archived on 2026-11-24 from an existing local note; this is not a new execution.
References to then-current documentation and test counts describe that historical
work. The source commit was not recorded. See [evidence scope](README.md).

## 2026-11-16：DGX Spark / Edge eager 冒烟

在 NVIDIA GB10（SM 12.1）、Linux ARM64、驱动 580.95.05 上使用项目独立的 uv 环境。
Python 3.13.9，PyTorch 2.14.1+cu130，torchvision 0.29.1+cu130，cuDNN 9.24.0.43。
默认推理入口先设置 `COSMOS_TRAINING=0`，避免引入训练用存储依赖；vendor 快照未修改。

- Policy：`nvidia/Cosmos3-Edge-Policy-DROID@a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14`。
- 原生 VAE：`Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e` 的 `Wan2.2_VAE.pth`，位于标准 Hugging Face 缓存。
- 输入：NVIDIA/cosmos-framework `e40831fa068fd171894b6cfb269308b441e80a85` 的 DROID 小样本，三路首帧、对应关节/夹爪状态和原指令。
- `uv sync --locked --extra edge` 成功；84 项测试通过，ruff 和 1357 个来源文件校验通过。
- 新环境中的原生 cuDNN BF16 attention / 主 LSE 与 FP32 参考数值检查通过。
- 真实模型 eager 完成 1 组预热及 1 组测量，共 4 次请求；随后设置 `HF_HUB_OFFLINE=1` 使用本地资产复跑成功。
- 每次 Sparse 均完成 8 个 forward、224 个 block 调用、7 次 pack/restore、4 次 velocity 更新；网格 17×20，每帧保留 184 tokens，GEN 从 3093 缩至 1845。
- Dense/Sparse 动作均为有限的 `[32, 8]`，vision latent 有限。峰值 allocated 8.242 GiB，reserved 8.461 GiB。

最终离线复跑的单组测量：Dense 2.5247 s，Sparse 1.6644 s，action MSE 0.017466。
计时覆盖预处理、生成及动作后处理；一次测量仅作冒烟记录，不设速度或 MSE 门槛，也不代表闭环成功率。
完整报告为 `outputs/smoke/edge.json`，日志为 `outputs/smoke/edge.log`；复跑命令见 README 的 Spark smoke test。
本轮覆盖 Edge eager，Nano、compile 与 CUDA Graph 未作新的实机验收。

首次下载时本机代理传输较慢，使用命令级 `env -u http_proxy -u https_proxy`；模型下载采用 `HF_HUB_DISABLE_XET=1` 的 HTTP 路径，未修改全局代理配置。

提交前复查修复了样本准备工具的自定义输出路径问题：无 `.npz` 后缀时，NumPy 原先自动追加后缀，导致后续来源哈希读取失败；现在通过文件句柄写入指定路径，并覆盖三种后缀情况。README 同时明确区分生成的配置摘要与原生 smoke 覆盖配置。

复查结果：86 项测试、ruff、锁文件校验、来源校验、五个 CLI 帮助入口、源码包及 wheel 构建通过。待提交文件中未发现模型/数据二进制、大于 5 MiB 的文件或常见凭据格式；虚拟环境、checkpoint 和输出报告均被 Git 忽略。构建使用的 setuptools 对原有 `project.license` 表格格式发出弃用提示，当前构建成功。

手动复跑报告 `outputs/smoke/edge_manual.json`（启动于 2026-11-16 19:04:34，UTC+8）同样通过四次 Dense/Sparse 检查，动作形状和稀疏执行计数符合预期。该次测量 Dense 2.5515 s、Sparse 1.6101 s，action MSE 0.017466；仍仅作单组冒烟记录。

已知环境限制：官方 `nvidia-cusparselt-cu13==0.8.1` ARM64 wheel 的内部平台标签是 `manylinux2014_sbsa`，因此 `uv pip check --python .venv/bin/python` 报一个平台不匹配，`uv sync` 会从缓存重新安装该包。上述真实 GPU 冒烟已在此包下通过；未修改官方 wheel 或安装元数据。
