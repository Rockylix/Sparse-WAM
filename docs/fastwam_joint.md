# FastWAM-Joint on LIBERO

FastWAM-Joint uses the same `create_service(...).infer(observation)` interface as Cosmos3 Edge.
It takes one raw LIBERO observation and returns a NumPy array of **32 seven-dimensional
actions** in simulator units. The K56 sparse policy retains 19 frame-specific tokens and
37 shared tokens per future frame. Use `mode="dense"` for the original ten-step eager sampler.

## Setup

From the full source checkout on Linux with a CUDA 13-compatible driver and an RTX 4090:

```bash
uv sync --locked --extra fastwam-joint
source .venv/bin/activate
```

To use Edge in the same environment, add `--extra edge` to the sync command.
Supply the following local assets in [the native configuration](../configs/smoke_fastwam_joint.json).
The repository does not include model weights or task embeddings.

| JSON field | Local asset |
|---|---|
| `checkpoint_path` | FastWAM-Joint LIBERO checkpoint (`step_040000.pt`) |
| `dataset_stats_path` | Matching state/action normalization statistics (`dataset_stats.json`) |
| `vae_path` | [Wan 2.2 TI2V 5B VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/Wan2.2_VAE.pth) |
| `text_cache_dir` | Directory of cached T5 embeddings for the exact LIBERO task instructions you use |

Paths inside the JSON are relative to **that JSON file**. An explicit `checkpoint=`
argument overrides `checkpoint_path` and resolves relative to the current working directory.
The backend reads the cached embedding for each instruction; it does not run a text encoder.

## Run one observation

Use a recorded LIBERO observation dictionary with these fields:

| Field | Input |
|---|---|
| `agentview_image`, `robot0_eye_in_hand_image` | Unrotated RGB images, `uint8`, each `[H, W, 3]` |
| `robot0_eef_pos`, `robot0_eef_quat` | End-effector position `[3]` and quaternion `[x, y, z, w]` |
| `robot0_gripper_qpos` | Two gripper joint positions |
| `task_description` (or `prompt`) | The raw instruction string matching the cached embedding |

```python
from sparse_wam import SparseWAMConfig
from sparse_wam.service import create_service

service, _ = create_service(
    SparseWAMConfig("fastwam-joint", execution="eager"),
    native_config="configs/smoke_fastwam_joint.json",
    mode="sparse",
    seed=42,
)
actions = service.infer(observation)["action"]  # [32, 7]
```

The service rotates and resizes the cameras, normalizes state, and converts predicted
actions to LIBERO units. For repeated calls, reuse the service. Sparse inference also
accepts `execution="compile"` and `"compile-graph"`; both require a working C++
compiler. With `"compile-graph"`, keep all requests on the same CUDA stream.
