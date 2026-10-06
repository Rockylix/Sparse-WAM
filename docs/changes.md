# 相对来源分支的修改

1. 两个方法分支分别固定在 `SOURCE_MANIFEST.json` 中的 SHA；原 checkout 保持只读，不混入 main。
2. Core80/Stable104、Top6、动作权重、pool 和 Stable CV 规则统一到 `src/sparse_wam`。
3. 稀疏调度保持一次 conditional dense、七次 sparse；新增共享 controller 与请求级清理。
4. Edge 使用现有主-LSE ABI；vendor 的 `edge_core_stable_fast.py` 在传入主 LSE 时转交公共 scorer。
   原 fallback 留在 runtime 快照里供其内部旧入口兼容；项目 SparseWAMRunner 始终启用主 LSE。
5. Nano 从 ASI adapter 提取 GEN QKV/MLP 执行，主 attention 同时返回 LSE，不再额外执行 action-row attention。
6. 接入背景 velocity cache：由用户指定 conditional-only full-step 调度下的 full conditional 背景初始化，
   在 CFG 后、UniPC 前应用公式（7）。这是相对两个方法分支的行为变化。
7. 新增统一 Dense/Sparse RoboLab 服务接线。Dense 与 Sparse 均先正确序列化 dict caption。
8. 参数集中在 `SparseWAMConfig`；`configs/*.json` 是导出记录，不是另一套可漂移的默认值。
9. compiler/Graph 是显式执行选项，默认 eager；不声称编译保持逐位输出或成功率。
10. 保留原生 framework runtime 快照，而不进行未经 GPU 回归的版本覆盖或依赖激进删除。
11. 4090 冒烟修复：Nano 空间 token 数从当前 packed layout 读取，不再固定为 360；
    Core80+Stable104 的预算与选择公式不变，仍检查每个 chunk 内的布局一致性。
12. 配对 benchmark 增加 `--dense-eager`，隔离 Edge Dense 未编译路径与 Sparse 编译路径，
    并记录真实 CUDA Graph replay 次数。该切换仅用于 benchmark，不改变默认服务路径。
13. Nano Sparse 增加请求内原始 packing 复用，默认开启，可用 `--no-nano-packing-cache` 关闭。
    范围限制为原生 `_get_velocity` 的单样本标量 timestep 路径；按文本/CFG/几何分开缓存。
    每步仍更新 noisy tokens 和 timestep，不复用 attention 输出，不跨 chunk 缓存。
    Dense 不启用、Edge 不改变；代码位于共享层，不修改 vendor 快照。

14. 公共接口统一为 `SparseWAMConfig`、`SparseWAMRunner`；同步 CLI、测试和 benchmark 引用。
    这是接口命名变更，原配置类与 runner 的旧导入名不再导出，外部脚本需同步更新。
    README 改为面向使用者的入口；方法说明移至 `docs/method.md`，来源与许可证记录保留。
    不改变评分、选区、缓存、模型数值运算或编译开关。
    55 项 CPU 测试、CLI help、配置导出、ruff 与来源校验通过；对修改后的 src/tools
    做了去除文档字符串并归一化命名后的 AST 比较，计算结构与变更前一致。
    此次没有重跑 GPU 测速；README 中延迟仍引用此前已记录的测量。

`tools/check_sources.py` 用来源 SHA256 与已登记修改后的 SHA256 检查所有复制文件。
新核心代码的派生来源也在 manifest 中记录。原提交历史、远端、模型权重与实验原始数据未复制。
