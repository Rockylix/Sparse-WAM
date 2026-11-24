# Edge and Nano runtime differences

Edge is the primary installation target; follow the [RTX 4090 guide](rtx4090.md).
The adapters share Core80/Stable104 selection, Top6 scoring, action-time alignment,
main-attention LSE scoring, CFG scheduling and the background velocity cache.

| Property | Cosmos3 Edge | Cosmos3 Nano |
|---|---|---|
| Generator layers / hidden width | 28 / 2048 | 36 / 4096 |
| MLP | Two linear layers, ReLU²; intermediate 9216 | Three matrices, SwiGLU; intermediate 12288 |
| Spatial tokens per frame | 340 (17×20) for the DROID fixture | `P`, read from the actual packed input layout |
| Full GEN length | 3093 | `9P + 33` |
| Sparse GEN length | 1845 | `P + 8×184 + 33` |
| Transformer execution | Native Edge decoder | Nano GEN adapter |
| Compile / Graph | Native layer compilation | GEN adapter compilation; Dense control remains eager |
| Model residency | Native Edge service | Reasoner prefill/offload, GEN resident, instruction KV cache |

The 33 action positions comprise one state/conditioning position and 32 predictions.
L0 remains full; UND text tokens are excluded from the GEN lengths above. The
retained fraction is `184/P`, so pruning ratio and runtime depend on the actual
checkpoint and input geometry. Keep token IDs and their position encodings aligned.

Both runtime snapshots use the module name `cosmos_framework`; load Edge and Nano
in separate processes and environments. Native instruction KV reuse across chunks
has a different lifecycle from request-local Sparse-WAM masks and velocity caches.

Nano requires matching framework/CUDA dependencies, checkpoint, VAE and tokenizer.
The root `edge` extra does not install a Nano environment or provide a one-command
Nano acceptance workflow. Nano enables Reasoner offload and instruction caching;
per-layer GEN offload is incompatible with that execution path.

For method semantics see [Method](method.md); for Edge checks see
[Validation](validation.md).
