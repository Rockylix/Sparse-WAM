# Third-party notices

The root [LICENSE](LICENSE) is the default license for Sparse-WAM. Use the table below to identify exceptions and upstream notices for individual files.

| Files or directory | License and attribution |
| --- | --- |
| Project-owned files, including the shared API in `src/sparse_wam/`, unless listed below | OpenMDW-1.1; see the root `LICENSE`. |
| `vendor/edge/` and `vendor/nano/` | Cosmos Framework code and adaptations. Each directory retains its upstream OpenMDW-1.1 `LICENSE` and individual file notices. |
| `src/sparse_wam/backends/fastwam_joint.py` and `src/sparse_wam/backends/libero.py` | Original Joint interface and LIBERO adapter: MIT, copyright (c) 2026 Rockylix. See the [MIT text](vendor/fastwam_joint/LICENSE). |
| `vendor/fastwam_joint/` and the corresponding source excerpts in `tools/fastwam_joint_runtime.patch` | The FastWAM-Joint snapshot includes Rockylix’s MIT license. See the provenance and Qi license note below; existing third-party notices are preserved. |

## Cosmos Framework

This project includes Cosmos Framework code and adaptations from:

- Rockylix/Cosmos3-Task-relevant-Inference, branch `version/core80-stable104-action-weighted`,
  commit `deb4af65000d06b7b13e30c6ebcc0d44e6c67921`.
- Rockylix/Cosmos3-nano-policy, branch `version/asi`,
  commit `a89509e1f54e60f0622741995efd5e2f3d6710d6`.

The original framework copyright notice is:

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The Cosmos source repositories distribute their materials under OpenMDW-1.1. A copy is
retained in `LICENSE`, `vendor/edge/LICENSE` and `vendor/nano/LICENSE`. Original file-level
notices, including third-party notices within vendored modules, remain in place.
NVIDIA’s copyright notice does not attribute local modifications to NVIDIA.

`SOURCE_MANIFEST.json` records source repositories, branches, commits, file hashes,
and documented local changes. Removing Git history or fork relationships does not
remove these source and license attributions. The project is not presented as an
official NVIDIA repository or product.

Model checkpoints, datasets, CUDA libraries, PyTorch and other installed dependencies
are external assets with their own applicable terms. No model weights are bundled.

## FastWAM-Joint

The BF16 runtime in `vendor/fastwam_joint` derives from Rockylix/Fastwam-joint,
branch `version/asi-k56`, based on commit
`88fed984c03ea963ee6cc24f7753c59c20908c2d`. The included K56 model and controller
use the reference working-tree revisions recorded in `SOURCE_MANIFEST.json`.
The manifest identifies the source and adapted file hashes, and records the
committed file hashes for working-tree inputs that differ from that commit.

The source checkout includes an MIT license with copyright (c) 2026 Rockylix,
reproduced in `vendor/fastwam_joint/LICENSE`. Its README describes Qi as
Apache-2.0. This discrepancy requires upstream clarification for redistribution.
The reference checkpoint card for `Mikuz12/Spec-Wam-libero` separately declares
Apache-2.0. The Joint snapshot contains source code and model configuration.
