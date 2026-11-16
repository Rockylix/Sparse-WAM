# Sparse-WAM

**Action-guided sparse inference for World Action Models (WAMs).**

Sparse-WAM reduces future-vision token computation in joint video–action denoising,
without training or changing model weights. It selects task-relevant tokens from
action attention, executes the Transformer on compact token sequences, and uses
a first-step velocity cache to update unselected regions.

[Method](docs/method.md) · [Quick start](#quick-start) ·
[Benchmarks](#benchmarks) · [Validation](docs/validation.md)

## Overview

Sparse-WAM combines full-compute step selection, Core/Stable token selection,
sparse execution, and cached latent updates. The four-step Cosmos example below
uses conditional and unconditional CFG branches; backbone-specific schedules
are listed under [Supported configuration](#supported-configuration).

1. **Select full-compute steps.** Use designated dense forwards to collect
   action-to-future relevance across layers. In this configuration, the first
   conditional forward provides the full-compute reference.
2. **Select tokens.** Combine frame-specific Core tokens with shared Stable
   token positions to form the retained region for each future frame.
3. **Execute sparsely.** Reuse this mask for the remaining seven forwards.
   Selected tokens retain their original position encodings; observation and
   action tokens are not pruned.
4. **Update the full latent.** Use current predictions inside the selected
   region and cached step-0 velocity outside it, before the native UniPC update.

The mask and velocity cache reset at every action chunk. This is approximate
inference: retaining action tokens does not guarantee unchanged action outputs.

### Efficient execution

- Reuse the main attention's log-sum-exp (LSE) for scoring, avoiding a second
  attention-output computation. Local action–future QK products are still computed.
- Batch relevance scoring across future frames and reuse token-layout metadata.
- Reduce the actual attention and MLP input length, rather than masking dense outputs.
- Support eager, Torch Compile, and compiler-managed CUDA Graph execution.
- On Nano, offload the Reasoner after instruction prefill, reuse instruction KV,
  and cache request-local denoising packing metadata.

## Supported configuration

| Setting | Cosmos3 Edge | Cosmos3 Nano | FastWAM-Joint (LIBERO) |
|---|---|---|---|
| Transformer layers | 28 | 36 | 30 per expert |
| Hidden width | 2048 | 4096 | Video: 3072; action: 1024 |
| Future latent frames | 8 | 8 | 2 |
| Predicted action horizon | 32 | 32 | 32 |
| Spatial tokens per latent frame | 340 (17×20) | Inferred from the input layout | 98 (7×14) |
| Retained future tokens per frame | 184 = Core80 + Stable104 | 184 = Core80 + Stable104 | K31 default: Core19 + Stable12; K56 preset: Core19 + Stable37 |
| Denoising | 4 steps, guidance 3, shift 5 | 4 steps, guidance 3, shift 5 | 10 steps, guidance 1, shift 5 |
| Full / sparse forwards per chunk | 1 / 7 | 1 / 7 | 1 / 9 |
| Implementation | Included here | Included here | Separate [FastWAM-Joint repository](https://github.com/Rockylix/Fastwam-joint) |

For Cosmos, only step-0 conditional is dense; step-0 unconditional is already sparse.
The first-step background CFG initialization and restoration rules are described
in [Method](docs/method.md).

FastWAM-Joint values refer to the two-camera LIBERO configuration
(`configs/model/fastwam_joint.yaml`, `configs/data/libero_2cam.yaml`, and
`configs/sim_libero.yaml` in its repository). K31 is the current Qi checkout's
default; K56 is a separate available preset, not an additional default.
With guidance 1, it uses one forward per denoising step and caches the observation
K/V after step 0. Its action-only path advances packed sparse video states;
full background velocity reconstruction is used when predicting video.
The Cosmos four-step/UniPC execution example does not describe that backend's
entire sampling implementation.

This checkout provides single-GPU, batch-one Cosmos policy inference with native
UniPC. Edge and Nano use separate runtimes and must run in separate processes.
Their budgets and denoising schedule are fixed in this release. FastWAM-Joint
is documented here for comparison; it is not a `--backbone` option in this CLI.

## Quick start

### 1. Environment

Clone the complete source checkout:

```bash
git clone https://github.com/Rockylix/Sparse-WAM.git
cd Sparse-WAM
```

The commands below target RTX 4090 on Linux x86_64 with a CUDA 13-compatible
NVIDIA driver. The project environment is managed by
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --extra edge
uv run --locked --extra edge python -m sparse_wam config --backbone edge
```

`.python-version` selects Python 3.13. The Linux x86_64 lock pins PyTorch
2.14.1+cu130, torchvision 0.29.1+cu130 and cuDNN 9.24.0.43. Keep `--extra edge`
on subsequent `uv run` commands to select the inference dependencies.
Development tools are installed by the default `dev` group. The recorded RTX
4090 results use the earlier PyTorch 2.10.0+cu130 environment; the current locked
stack still requires a checkpoint-level smoke run on RTX 4090.

The pinned runtime requires Python >=3.12. Its older full training dependency
groups pin cuDNN 9.15, while its current attention backend requires >=9.22;
use the root project environment for this Edge eager workflow. Runtime activation
defaults `COSMOS_TRAINING=0` before loading the framework. It uses native cuDNN
attention without FlashAttention, NATTEN or Transformer Engine.

Keep the complete source checkout available: a standalone wheel does not bundle
the vendored runtimes. Nano requires its matching environment in a separate
process; the root `edge` extra covers Edge offline inference.

### 2. Model assets

Provide a local policy checkpoint together with its matching VAE and tokenizer.
Weights and datasets are not included.

- RoboLab examples use Cosmos3-Edge-Policy-DROID or Cosmos3-Nano-Policy-DROID.
- Piper HDF5 evaluation requires the matching Piper checkpoint and data adapter;
  it is not interchangeable with DROID evaluation.
- Use `--native-config` for local asset paths and native service settings. Its
  JSON fields follow the selected runtime's `RobolabServerArgs`.

Inspect the effective Sparse-WAM configuration without loading weights:

```bash
uv run --locked --extra edge python -m sparse_wam config --backbone edge --execution compile-graph
uv run --locked --extra edge python -m sparse_wam config --backbone nano --execution compile-graph
```

### 3. Start a policy server

WebSocket serving additionally requires a compatible OpenPI transport and any
auxiliary models enabled in the native JSON config. The minimal `edge` extra
covers offline inference; this server example assumes an existing compatible
backend environment with the editable Sparse-WAM package installed:

```bash
python -m sparse_wam serve \
  --backbone edge \
  --checkpoint-path /path/to/Cosmos3-Edge-Policy-DROID \
  --native-config /path/to/native.json \
  --mode sparse --execution compile-graph \
  --host 127.0.0.1 --port 8000 --seed 0
```

For Nano, set `--backbone nano` and use its checkpoint/environment.
Nano automatically enables Reasoner offload and instruction KV caching; GEN
weights remain on the GPU. Do not combine this with per-layer model offload.

To run the dense control, explicitly set `--mode dense --execution eager`.
The default execution backend is eager. CUDA Graphs are managed per compiled
layer/function, not as one graph around the complete sampling loop.
Nano packing reuse can be disabled with `--no-nano-packing-cache`.

### Smoke test (Edge)

Prepare the official checkpoint and one recorded NVIDIA DROID observation:

```bash
uv run --locked --extra edge python tools/prepare_smoke.py
uv run --locked --extra edge python tools/benchmark_policy.py \
  --backbone edge --checkpoint-path checkpoints/edge \
  --observation outputs/smoke/observation.npz \
  --native-config configs/smoke_edge.json --smoke \
  --output outputs/smoke/edge.json
```

The preparation tool records the resolved model and sample revisions in
provenance sidecars. The Edge checkpoint stays under `checkpoints/`; its pinned
native Wan2.2 VAE uses the standard Hugging Face cache (`HF_HOME` is respected).
Samples, logs and reports stay under `outputs/`. Both project directories are
ignored by Git. Download access may
require `HF_TOKEN` with access to the NVIDIA model repositories.

`--smoke` runs eager inference with one warmup pair and one measured pair,
using the same input and seed for Dense/Sparse in each pair. It checks finite
`[32, 8]` actions and the actual sparse execution counters, then records versions,
provenance, timings, memory use and action MSE in JSON. A failed check exits
nonzero. The fixture config skips auxiliary content guardrail models and video
decoding for this offline test; serving defaults are unchanged. No speedup or
closed-loop task-success threshold is imposed.

## Benchmarks

### Recorded single-chunk latency

Synchronized, warm single-chunk latency on RTX 4090:

| Model / input | Before optimization | After optimization | Speedup |
|---|---:|---:|---:|
| Edge DROID / Banana observation, 340 tokens per frame | 0.8521 s | 0.4615 s | 1.846× |
| Nano Piper20k / HDF5, 360 tokens per frame | 3.3409 s | 1.8388 s | 1.817× |
| FastWAM-Joint / LIBERO, K56 (Core19 + Stable37) | 0.4273 s | 0.2151 s | 1.986× |

Before optimization: **Dense eager**. After optimization: **sparse + Torch Compile + CUDA Graph**.
Values are medians of warm measurements, not cold-start times.
Edge uses 5 measured pairs; Nano uses 20; both follow 3 warmup pairs.
These measure **sparsity plus compilation**, not the isolated effect of pruning.
Cosmos rows time generation only; the FastWAM-Joint row times its native
action-only inference call with the K56 preset (not the K31 default).
The latter is an existing [FastWAM-Joint benchmark record](https://github.com/Rockylix/Fastwam-joint/blob/88fed98/docs/fastwam_joint_sparse_cn.md),
not a new measurement through this repository's CLI. Backend timing scopes differ;
each speedup compares that backend against its own dense control.
Cold compilation/prefill, external input preprocessing, RPC and future-video
decoding are excluded. These timings do not establish closed-loop task success.
See [validation details](docs/validation.md) for configurations and output checks.

### Paired observation benchmark

Provide an NPZ containing a recorded RoboLab/openpi observation. Keys must match
the native service input; arrays are NumPy values and `prompt` is a Unicode scalar.
Pickled objects are not accepted.

```bash
uv run --locked --extra edge python tools/benchmark_policy.py \
  --backbone edge \
  --checkpoint-path /path/to/Cosmos3-Edge-Policy-DROID \
  --native-config /path/to/native.json \
  --observation /path/to/observation.npz \
  --execution compile-graph --dense-eager \
  --warmup 3 --repeats 20 --seed 0 \
  --output outputs/edge_benchmark.json
```

This tool measures **complete policy inference**, including preprocessing and
action postprocessing, so its speedup need not match the generation-only table.
It reports mean/median/P90, paired action MSE, token counts and observed Graph replays.

Additional tools:

- `tools/benchmark_nano_hdf5.py`: generation-only Piper HDF5 evaluation using
  `SparseWAMRunner`; requires an external Piper checkout for data reading/collation.
- `tools/benchmark_nano_packing.py`: paired packing on/off timing and output regression.

Run either tool with `--help` for required inputs and options.

## Development

```bash
uv run --locked --extra edge pytest -q
uv run --locked --extra edge ruff check src tests tools
uv run --locked --extra edge python tools/check_sources.py
```

Tests cover scoring, token selection, sparse execution contracts, cache lifecycle
and benchmark dispatch. CPU tests do not substitute for checkpoint-level GPU or
closed-loop evaluation.

| Path | Purpose |
|---|---|
| `src/sparse_wam/` | Selection, scoring, packing, velocity cache and runtime |
| `src/sparse_wam/backends/` | Edge and Nano execution adapters |
| `configs/` | Configuration summaries and native smoke overrides |
| `tools/` | Benchmarks and source verification |
| `tests/` | Unit and numerical tests |
| `vendor/` | Pinned Cosmos runtimes |
| `docs/` | Method, backend details and validation records |

Python entry points are `SparseWAMConfig` in `sparse_wam.config` and
`SparseWAMRunner` in `sparse_wam.runtime`. `configs/edge.json` and
`configs/nano.json` are generated configuration summaries. `configs/smoke_edge.json`
provides native runtime overrides for the offline smoke test.

## License and acknowledgments

Distributed under [OpenMDW-1.1](LICENSE). Sparse-WAM builds on Cosmos Framework;
third-party notices and source provenance are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). This is not an official NVIDIA release.
Model weights and external dependencies remain subject to their respective terms.
