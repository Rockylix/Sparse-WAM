# Archived validation evidence

## 2026-11-24 release validation

The [final paired smoke](edge-gb10-20261124.json) and
[standalone example summary](edge-gb10-20261124-example.json) were produced from
clean source commit `168f04175487c9361fe7cd2de4b102164461de08`, with implementation
SHA256 `5ea9d51e6fdd22b7807769ee5d4397d1489c95fee0200e6223c832b876c2c7dc`.
Both reports record `git_dirty=false`, the command, native configuration hash and
observation hash. The [release manifest](release-20261124-manifest.json) records
source/archived file hashes and path redactions separately from the historical
archive below. The example summary has no original timestamp; none was invented.

The paired run started on 2026-11-24 at 11:48:25 UTC. It used NVIDIA GB10, Linux
ARM64, Python 3.13.9, PyTorch 2.14.1+cu130 and cuDNN 9.24.0.43. One warmup pair and
one measured Dense/Sparse pair passed all four request checks. Actions were finite
`[32, 8]`; Sparse completed one dense and seven sparse forwards, seven pack/restore
pairs and four velocity updates on the expected 17×20/K184 layout.

The measured full-policy times were 2.5248 seconds Dense and 1.6376 seconds Sparse,
with action MSE 0.017466 and peak allocated/reserved memory of 8.242/8.461 GiB.
These single-pair measurements are functional smoke evidence, not a performance
distribution or action-quality acceptance threshold. The standalone example also
produced `[32, 8]` actions and the expected sparse execution summary; only its JSON
summary is archived, not the action NPZ.

This validates Edge eager on the recorded GB10 environment. It adds no RTX 4090,
Nano, compile/CUDA Graph or closed-loop validation. These documentation/evidence
additions follow the tested code commit and do not alter its implementation digest.

## 2026-11-16 historical archive

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
