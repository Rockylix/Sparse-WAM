# Python API and input contracts

Use an editable installation from the complete source checkout. The vendored
`cosmos_framework` runtimes are loaded from that checkout; a standalone wheel
does not include them. Run the examples from the repository root, or use absolute
paths for checkpoints, native JSON configuration and observation files.

## Public entry points

```python
from sparse_wam import SparseWAMConfig
from sparse_wam.runtime import SparseWAMRunner
from sparse_wam.service import create_service
```

| Entry point | Input | Output / responsibility |
|---|---|---|
| `SparseWAMConfig(backbone, execution="eager", nano_packing_cache=True)` | `"edge"`, `"nano"` or `"fastwam-joint"` with execution `"eager"`, `"compile"` or `"compile-graph"` | Immutable method settings; `to_dict()` exports a configuration summary |
| `create_service(config, ...)` | Model checkpoint and optional native configuration | `(service, args)`; loads the model and installs the runner into native generation |
| `service.infer(observation)` | One recorded DROID (Cosmos) or LIBERO (Joint) observation dictionary | Dictionary containing a NumPy `action` array and optionally decoded `video` |
| `SparseWAMRunner(model, config, mode="sparse", generate=None)` | An initialized native Cosmos model | Callable wrapper around native `generate_samples_from_batch` |
| `runner(data_batch, **kwargs)` | One native, transformed/collated policy batch | Native output dictionary, including one-element `action` and `vision` tensor lists |

`SparseWAMConfig` does not load weights. `config.to_dict()` describes the fixed
method settings; it is not a native server configuration. The files
`configs/edge.json`, `configs/nano.json` and `configs/fastwam_joint.json` are exported summaries, not inputs to
`--native-config`. Use `run_metadata("dense")` or `run_metadata("sparse")` for
mode-specific metadata and `runner.last_summary` for the most recently completed
request's actual execution summary.

For the LIBERO observation fields and Joint asset paths, see [FastWAM-Joint](fastwam_joint.md).

## Offline Edge inference

The runnable [example](../examples/infer_edge.py) loads an NPZ observation, invokes
the same public service used by the optional WebSocket entry point, and saves the
action and summary. With assets prepared as described in the [RTX 4090 guide](rtx4090.md):

```python
import numpy as np

from sparse_wam import SparseWAMConfig
from sparse_wam.service import create_service

config = SparseWAMConfig("edge", execution="eager")
service, args = create_service(
    config,
    checkpoint="checkpoints/edge",
    native_config="configs/smoke_edge.json",
    mode="sparse",
    seed=0,
)
with np.load("outputs/smoke/observation.npz", allow_pickle=False) as source:
    observation = {
        key: source[key].item() if source[key].ndim == 0 else source[key].copy()
        for key in source.files
    }
result = service.infer(observation)
action = result["action"]
summary = service.sparse_wam_runner.last_summary
```

`create_service` accepts keyword arguments `checkpoint=None`, `host="127.0.0.1"`,
`port=8000`, `seed=0`, `output_dir="outputs/server"`, `mode="sparse"`, and
`native_config=None`. Creating a service loads assets but does not open a network
listener. `mode` is `"dense"` or `"sparse"`.

Checkpoint selection is: explicit `checkpoint` / CLI `--checkpoint-path`, then
`checkpoint_path` from the native JSON, then the selected backbone's default.
The native JSON is an object whose fields follow the selected runtime's
`RobolabServerArgs`. Invalid fields or incompatible sampling options fail rather
than changing the reproduction protocol. Host, port, seed and output directory
are controlled by the function/CLI arguments, including their defaults.

`configs/smoke_edge.json` disables auxiliary guardrail models and video decoding
for offline testing. A different deployment can supply its own asset and service
configuration. Relative paths in arguments and native JSON refer to the process
working directory, not the directory containing the JSON file.

## DROID observation and result

The default released Edge DROID configuration uses the following contract:

| Dictionary key | Value |
|---|---|
| `prompt` | Nonempty task instruction string |
| `observation/image` | Composed RGB image, NumPy `uint8`, `[H, W, 3]` |
| `observation/joint_position` | Seven joint positions, floating-point `[7]` or `[T, 7]` |
| `observation/gripper_position` | Floating-point scalar, `[T]` or `[T, 1]`, using the native RoboLab/DROID convention |

Instead of `observation/image`, provide all three RGB arrays:
`observation/wrist_image_left`, `observation/exterior_image_1_left`, and
`observation/exterior_image_2_left`. The native preprocessing combines the wrist
view with the two resized exterior views. If `observation/image` is present, it
takes precedence. The native transform handles resizing and checkpoint-specific
normalization; do not feed an already normalized latent tensor as an image.

NPZ observations store these same keys. Store `prompt` as a NumPy Unicode scalar;
object/pickled arrays are not accepted by the example or benchmark. Use finite
state values and a sample compatible with the selected checkpoint and domain.
The prepared fixture contains a composed image and one state observation.

For the default Edge DROID settings, `result["action"]` is `[32, 8]`: 32 predicted
steps, seven joint positions and one gripper value. The service performs the native
history-row removal and gripper conversion. When `decode_video=True`, `video` is
also returned as RGB `uint8` `[T, H, W, 3]`. Decoding adds work and is excluded from
the recorded latency comparisons. Other native domains/checkpoints have their own
action semantics and require matching adapters; Piper data is not interchangeable
with the DROID fixture.

## Integrating an existing native model

Use `SparseWAMRunner` when a caller already owns model loading and native batch
preparation:

```python
runner = SparseWAMRunner(model, SparseWAMConfig("edge"), mode="sparse")
samples = runner(data_batch, guidance=3.0, num_steps=4, shift=5.0)
summary = runner.last_summary
```

Here `model` is a fully initialized model from the matching vendored runtime and
`data_batch` is the native policy batch after its transforms and collation. It is
not the observation dictionary accepted by `service.infer`. Preserve the native
model's keys, tensor devices, layout and conditioning metadata. The batch must
contain exactly one caption under `model.input_caption_key`; the runner serializes
a structured dictionary caption as JSON before tokenization.

The runner returns native `action` and `vision` lists, each containing one nonempty,
finite tensor. Vision is a latent tensor at this level. Raw action tensors may
include state/history rows and padded dimensions; they are not the postprocessed
`[32, 8]` service result. The runner does not perform DROID observation preprocessing,
model loading, action conversion or video decoding.

Wrap generation once. `create_service` already installs a runner as
`service.model.generate_samples_from_batch`; do not wrap that model a second time.
The optional `generate` argument is an unwrapped native generation callable for
integrators that need to supply it explicitly.

## Execution and lifecycle

- This release uses one GPU/rank and one policy sample, native UniPC, four denoising
  steps, guidance 3 and shift 5. Sparse execution retains Core80 + Stable104 future
  tokens per frame, with one dense conditional forward and seven sparse forwards.
- Run calls serially for each model/runner and read `last_summary` immediately after
  the successful call. Runner hooks, sampler wrappers, model state and summaries
  are mutable; the runner is not a concurrent request interface. Use independent
  worker processes for independent model instances.
- Selection masks, background velocity and Nano packing metadata are request-local.
  The native instruction KV cache has a different lifecycle. Successful inference
  updates `last_summary`; a failed inference must not be interpreted as a successful
  result or reuse an earlier summary.
- Edge and Nano use different snapshots with the same `cosmos_framework` module
  name. Load them in separate processes and environments. Activating one after the
  other in the same process is rejected.
- Eager is the default. Compile and compiler-managed CUDA Graphs require separate
  target-device validation and warmup. No single graph encloses the whole sampler.
  Nano automatically enables Reasoner offload/instruction KV caching; GEN stays
  resident and per-layer offload is incompatible.
- Unsupported custom samplers, CFG schedules, prompt upsampling and multi-rank
  execution are outside this release. See [method constraints](method.md#运行约束).

## CLI and optional transport

`python -m sparse_wam config --backbone edge` prints method configuration without
loading a checkpoint. `python -m sparse_wam serve --help` lists the optional
WebSocket entry point. It uses the external OpenPI transport and the same service;
no REST API is defined by this package. Offline `create_service` inference does
not require the transport. See the [server example](rtx4090.md#optional-websocket-service)
for the required compatible backend environment.

## Benchmark provenance

The offline example and benchmark reports include a `reproduction` object, captured
before inference timing. It supplements model/sample provenance and runtime
environment details; it does not establish numerical correctness by itself.

| Field | Meaning |
|---|---|
| `source.version` | Installed Sparse-WAM package version |
| `source.git_commit`, `source.git_dirty`, `source.git_status` | Commit and working-tree state of this checkout; `null` if Git metadata is unavailable |
| `source.implementation_sha256` | Digest of a sorted mapping of implementation paths to their file SHA256 hashes |
| `source.manifest_sha256` | SHA256 of `SOURCE_MANIFEST.json`, recorded separately |
| `argv` | Invocation argument list |
| `native_config` | Native JSON path and file SHA256, or `null` when no file was supplied |
| `input_files` | Named input file paths and SHA256 hashes, including the observation where applicable |

The implementation digest covers Python files under `src/sparse_wam/`, `tools/`
and `examples/`, JSON files under `configs/`, and `pyproject.toml`, `uv.lock` and
`.python-version`. It excludes documentation, output artifacts, weights and vendor
contents. Paths use their repository-relative names in the sorted digest mapping.
Uncommitted implementation edits therefore change the digest even when the Git
commit is unchanged.

The manifest digest identifies the manifest file; it does not validate every
vendored file. Run `python tools/check_sources.py` separately to check the recorded
source hashes and registered modifications. A clean Git status is likewise not a
substitute for the model-specific execution checks.

A source archive without `.git` records Git fields as `null`, without borrowing an
enclosing repository's commit. Missing optional files have a `null` hash; ordinary
input validation still determines whether inference can run. Reports that lack
these fields remain incomplete records and must not have a
current commit inserted retroactively. Review local paths and command arguments
before publishing reports, and preserve the original report hash when redacting.
