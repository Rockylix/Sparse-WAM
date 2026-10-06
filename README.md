# Sparse-WAM

**Action-guided sparse inference for Cosmos3 Edge and Nano policies.**

Sparse-WAM reduces future-vision token computation in joint video–action denoising,
without training or changing model weights. It selects task-relevant tokens from
action attention, executes the Transformer on compact token sequences, and uses
a first-step velocity cache to update unselected regions.

[Method](docs/method.md) · [Quick start](#quick-start) ·
[Benchmarks](#benchmarks) · [Validation](docs/validation.md)

## Overview

Each action chunk uses four denoising steps with conditional and unconditional
CFG branches:

1. **Score once.** Run the first conditional forward densely and collect
   action-to-future relevance across layers.
2. **Select tokens.** Combine 80 frame-specific Core tokens with 104 shared
   Stable positions per future frame.
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

| Setting | Edge | Nano |
|---|---|---|
| Transformer layers | 28 | 36 |
| Hidden width | 2048 | 4096 |
| Future latent frames | 8 | 8 |
| Predicted action horizon | 32 | 32 |
| Spatial tokens per latent frame | 340 | Inferred from the input layout |
| Retained future tokens per frame | 184 = Core80 + Stable104 | 184 = Core80 + Stable104 |
| Denoising | 4 steps, guidance 3, shift 5 | 4 steps, guidance 3, shift 5 |

Only step-0 conditional is dense; step-0 unconditional is already sparse.
The first-step background CFG initialization and restoration rules are described
in [Method](docs/method.md).

Current support is single-GPU, batch-one policy inference with the native UniPC
sampler. Edge and Nano use separate runtimes and must run in separate processes.
Budgets and the denoising schedule are fixed in this release.

## Quick start

### 1. Environment

Clone the complete source checkout:

```bash
git clone https://github.com/Rockylix/Sparse-WAM.git
cd Sparse-WAM
```

Use a compatible Cosmos environment with Linux, CUDA, PyTorch and the matching
attention kernels. The recorded GPU tests use an RTX 4090 and PyTorch 2.10.0+cu130.
In an existing compatible environment, install only the Sparse-WAM package:

```bash
python -m pip install --no-deps -e .
```

For a fresh environment, dependency manifests are provided under
`vendor/edge` and `vendor/nano`. An example CUDA 13 setup for Edge is:

```bash
uv sync --project vendor/edge --all-extras --group cu130-train --group policy-server
uv pip install --python vendor/edge/.venv/bin/python --no-deps -e .
source vendor/edge/.venv/bin/activate
```

For Nano, use `vendor/nano` instead. This is the full upstream dependency set,
not a minimal installation; fresh installation has not been independently
validated on every platform. Keep the source checkout available: a standalone
wheel does not bundle the vendored runtimes.

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
python -m sparse_wam config --backbone edge --execution compile-graph
python -m sparse_wam config --backbone nano --execution compile-graph
```

### 3. Start a policy server

Run the following command in the corresponding backend environment:

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

## Benchmarks

### Recorded single-chunk latency

Synchronized, warm **generation-only** latency on RTX 4090:

| Model / input | Dense eager median | Sparse compile + Graph median | Speedup |
|---|---:|---:|---:|
| Edge DROID / Banana observation, 340 tokens per frame | 0.8521 s | 0.4615 s | 1.846× |
| Nano Piper20k / HDF5, 360 tokens per frame | 3.3409 s | 1.8388 s | 1.817× |

Edge uses 5 measured pairs; Nano uses 20; both follow 3 warmup pairs.
These measure **sparsity plus compilation**, not the isolated effect of pruning.
They exclude cold compilation/prefill, external input preprocessing, RPC and
future-video decoding. They do not establish closed-loop task success.
See [validation details](docs/validation.md) for configurations and output checks.

### Paired observation benchmark

Provide an NPZ containing a recorded RoboLab/openpi observation. Keys must match
the native service input; arrays are NumPy values and `prompt` is a Unicode scalar.
Pickled objects are not accepted.

```bash
python tools/benchmark_policy.py \
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
python -m pytest -q
python -m ruff check src tests tools
python tools/check_sources.py
```

Tests cover scoring, token selection, sparse execution contracts, cache lifecycle
and benchmark dispatch. CPU tests do not substitute for checkpoint-level GPU or
closed-loop evaluation.

| Path | Purpose |
|---|---|
| `src/sparse_wam/` | Selection, scoring, packing, velocity cache and runtime |
| `src/sparse_wam/backends/` | Edge and Nano execution adapters |
| `configs/` | Exported configuration summaries |
| `tools/` | Benchmarks and source verification |
| `tests/` | Unit and numerical tests |
| `vendor/` | Pinned Cosmos runtimes |
| `docs/` | Method, backend details and validation records |

Python entry points are `SparseWAMConfig` in `sparse_wam.config` and
`SparseWAMRunner` in `sparse_wam.runtime`. Configuration summaries in
`configs/` are generated records, not runtime override files.

## License and acknowledgments

Distributed under [OpenMDW-1.1](LICENSE). Sparse-WAM builds on Cosmos Framework;
third-party notices and source provenance are retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). This is not an official NVIDIA release.
Model weights and external dependencies remain subject to their respective terms.
