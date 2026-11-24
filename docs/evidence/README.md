# Archived validation evidence

These are existing local records from 2026-11-16, archived on 2026-11-24. Archiving
does not rerun inference or validate the current source revision. The original
reports did not record a Sparse-WAM commit or dirty status, so that association
remains unknown. The [manifest](manifest.json) records original and archived SHA256
hashes, source locations and redactions. Machine-specific checkout/cache paths use
`<CHECKOUT>` and `<HF_CACHE>`; measurement values and asset revisions are unchanged.

| File | Original run | What it supports |
|---|---|---|
| [Edge smoke](edge-gb10-20261116.json) | 2026-11-16 07:30:33 UTC | Successful Edge eager paired functional smoke on GB10 |
| [Manual rerun](edge-gb10-20261116-manual.json) | 2026-11-16 11:04:34 UTC | A second successful paired smoke with the same recorded assets |
| [Historical local note](spark-validation-20261116.md) | 2026-11-16 | Contemporaneous setup, checks and environment limitations; not an independent run report |

Both JSON reports record Python 3.13.9, PyTorch 2.14.1+cu130, cuDNN 9.24.0.43,
NVIDIA GB10 (SM 12.1), Linux ARM64, the model/VAE/sample revisions and observation
hash. Each has one warmup pair and one measured pair. Sparse requests completed
eight forwards, seven pack/restore pairs and four velocity updates, retaining
184 future tokens per frame on a 17×20 grid. Dense and Sparse actions were finite
`[32, 8]` arrays. The reported action MSE was approximately 0.017466 in each run;
no action-quality threshold was applied.

The single measured pairs took 2.5247/1.6644 seconds and 2.5515/1.6101 seconds for
Dense/Sparse respectively, including preprocessing and action postprocessing.
These are smoke observations, not a benchmark distribution. They do not establish
closed-loop success, validate Nano, or test compile/CUDA Graph execution.

The current documented lock targets Linux x86_64. A historical ARM64 run with
matching package versions does not prove that this lock installs on ARM64. The
local note also records a cuSPARSELt wheel platform-tag mismatch reported by
`uv pip check`; that environment limitation is preserved in the archive.

The historical RTX 4090 tables in [validation](../validation.md) reference other
local reports whose originals are not present in this checkout. They are retained
as historical documentation, explicitly separate from the archived GB10 evidence.
