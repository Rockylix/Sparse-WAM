# RTX 4090 validation

Use this checklist with the [Edge installation guide](rtx4090.md) on Linux x86_64
and an RTX 4090. It defines checks to run against your checkout and environment;
it is not a record of completed target-device testing.

## Software checks

From the repository root after installing the Edge environment:

```bash
uv run --locked --extra edge pytest -q
uv run --locked --extra edge ruff check src tests tools examples
uv run --locked --extra edge python tools/check_sources.py
uv run --locked --extra edge python -m sparse_wam config --backbone edge
uv run --locked --extra edge python examples/infer_edge.py --help
uv build --out-dir dist
uv run --locked --extra edge python tools/check_distribution.py --dist dist
```

All commands must exit successfully. Source verification checks the pinned vendor
hashes and registered modifications. Distribution checks exercise the unpacked
source package as well as the wheel's contents. These software checks do not
replace inference with the real checkpoint.

## Model smoke

After [preparing the model assets and observation](rtx4090.md#model-assets-and-recorded-input):

```bash
uv run --locked --extra edge python tools/benchmark_policy.py \
  --backbone edge --checkpoint-path checkpoints/edge \
  --observation outputs/smoke/observation.npz \
  --native-config configs/smoke_edge.json --smoke \
  --output outputs/smoke/edge.json
```

The tool runs one warmup pair and one measured pair. Dense and Sparse share the
same input and seed within each pair. Require a zero exit status and
`smoke.passed=true`; a failed or incomplete report is not a successful run.

| Check | Expected for the Edge DROID fixture |
|---|---|
| Device and assets | Report identifies the intended GPU, environment, checkpoint revision and observation hash |
| Action output | Finite `[32, 8]` arrays in Dense and Sparse |
| Dense execution | 8 dense forwards, 0 sparse forwards, 340 retained tokens per frame |
| Sparse execution | 1 dense and 7 sparse forwards; 8 completed forwards and 224 block calls |
| Sparse packing and updates | 7 pack/restore pairs, 4 velocity updates |
| Token geometry | 17×20 spatial grid; Core80 + Stable104 = 184 retained future tokens per frame; GEN length 3093 → 1845 |

Inspect `effective_runs`, `action_shapes`, `smoke.checks`, `environment`, `provenance`
and `reproduction` in the JSON report. The offline fixture disables video decoding
and auxiliary guardrail models. No speedup or action-MSE threshold is imposed.

## Further acceptance

- Run the [offline example](rtx4090.md#offline-inference) and inspect the saved action
  NPZ and JSON summary. Use different output paths to preserve earlier runs.
- Before integrating a persistent service, exercise different observations and
  instructions in successive requests. Check request-local mask/velocity updates,
  native instruction-cache invalidation and recovery after a failed request.
- Run [paired benchmarks](rtx4090.md#paired-benchmark) in eager first, then evaluate
  compile and compile-graph on the target device. Record actual Graph replays and
  compare output differences as well as latency; compilation need not be bitwise
  identical to eager execution.
- Keep the timing scope and controls explicit: full `service.infer` includes
  preprocessing and action postprocessing; Dense eager versus compiled Sparse
  measures sparsity and compilation together. Separate cold start from warm runs.
- Closed-loop task success requires the corresponding robot or simulator protocol.
  Finite outputs, action MSE and single-chunk latency do not establish that result.

Save each run under a distinct `outputs/` path with its source/configuration/input
metadata. See [benchmark provenance](api.md#benchmark-provenance) for the report
fields. Do not replace unknown provenance with the current checkout's values.
