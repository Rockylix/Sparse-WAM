# 相对来源分支的修改

1. 两个方法分支分别固定在 `SOURCE_MANIFEST.json` 中的 SHA；原 checkout 保持只读，不混入 main。
2. Core80/Stable104、Top6、动作权重、pool 和 Stable CV 规则统一到 `src/sparse_wam`。
3. 稀疏调度保持一次 conditional dense、七次 sparse；新增共享 controller 与请求级清理。
4. Edge 使用现有主-LSE ABI；vendor 的 `edge_core_stable_fast.py` 在传入主 LSE 时转交公共 scorer。
   原 fallback 留在 runtime 快照里供其内部旧入口兼容；项目 PaperRunner 始终启用主 LSE。
5. Nano 从 ASI adapter 提取 GEN QKV/MLP 执行，主 attention 同时返回 LSE，不再额外执行 action-row attention。
6. 接入背景 velocity cache：由用户指定 conditional-only full-step 调度下的 full conditional 背景初始化，
   在 CFG 后、UniPC 前应用公式（7）。这是相对两个方法分支的行为变化。
7. 新增统一 Dense/Sparse RoboLab 服务接线。Dense 与 Sparse 均先正确序列化 dict caption。
8. 参数集中在 `PaperConfig`；`configs/*.json` 是导出记录，不是另一套可漂移的默认值。
9. compiler/Graph 是显式执行选项，默认 eager；不声称编译保持逐位输出或成功率。
10. 保留原生 framework runtime 快照，而不进行未经 GPU 回归的版本覆盖或依赖激进删除。

`tools/check_sources.py` 用来源 SHA256 与已登记修改后的 SHA256 检查所有复制文件。
新核心代码的派生来源也在 manifest 中记录。原提交历史、远端、模型权重与实验原始数据未复制。
