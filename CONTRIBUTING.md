# Contributing

Sparse-WAM is a research implementation with Edge as the primary installation and
reproduction target. Keep changes to packaging, examples and interfaces separate
from changes to scoring, selection, CFG scheduling, packing or velocity updates.
Describe any numerical or lifecycle change explicitly in a pull request.

## Development checks

Use the complete source checkout and the [Linux x86_64 Edge environment](docs/rtx4090.md):

```bash
uv sync --locked --extra edge
uv run --locked --extra edge pytest -q
uv run --locked --extra edge ruff check src tests tools examples
uv run --locked --extra edge python tools/check_sources.py
git diff --check
```

The CPU test suite checks mathematical and lifecycle contracts; it does not load
production checkpoints or prove GPU performance. Automated checks should exercise
public imports, CLI/example help, configuration validation, tests, lint, source
provenance and source/wheel builds without downloading model weights. A successful
wheel build does not mean it is a standalone deployment: the vendored runtimes
require the complete editable checkout.

Add tests for a changed input contract, failure path or execution lifecycle. Test
missing optional components without requiring those components in the CPU job.
Preserve useful errors before model loading where the invalid input can be checked
without it. Update the [API documentation](docs/api.md) when interfaces change.

## Reproduction evidence

Run the [Edge smoke](docs/validation.md#model-smoke) on RTX 4090 after changes
that affect model loading or inference. Record the source version/commit and dirty
state, exact command/configuration, environment, checkpoint and observation
provenance, timing scope, execution counters and pass/fail status. Missing metadata
must remain explicitly unknown; do not infer it from the current checkout.

Use matched inputs and seeds for Dense/Sparse comparisons. Separate generation
latency from the full preprocessing/inference/postprocessing path, and separate
cold start from warmed inference. A one-pair smoke verifies execution and finite
outputs; it does not establish a speedup distribution, action-quality threshold or
closed-loop success rate. Label comparisons that combine sparsity and compilation.

Keep weights, datasets, transient logs and raw runs in the ignored `checkpoints/`
and `outputs/` directories. When sharing a run report for review, remove
machine-specific paths while preserving measurement values and asset revisions;
include the original-file hash and describe any redactions. Keep hardware-specific
run logs out of the public installation documentation. Never change a report's
recorded source revision to imply a different code version was tested.

## Source snapshots

`vendor/` and `src/sparse_wam/layout.py` contain pinned source files checked by
`tools/check_sources.py`. Retain original license notices and provenance. A required
snapshot change needs an explicit manifest entry, a reason and appropriate runtime
regression evidence; do not refresh vendor files as a side effect of formatting or
dependency work. Model weights and datasets remain separately obtained assets.
