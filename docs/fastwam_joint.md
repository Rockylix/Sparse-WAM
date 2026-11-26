# FastWAM-Joint on LIBERO

The offline backend takes one recorded LIBERO observation and returns 32
seven-dimensional actions in simulator units. Dense eager runs the original
joint video/action sampler with the same weights and ten denoising steps as
the sparse policy. Both emit actions without decoding predicted video. A service serializes model calls.
Use the capture CUDA stream for subsequent graph replays.

## Environment and local assets

Run from the full source checkout. Use the same project environment for Edge
and Joint, including the host C++ compiler used for TorchInductor.

```bash
uv sync --locked --extra edge --extra fastwam-joint
source .venv/bin/activate
python -m sparse_wam config --backbone fastwam-joint
```

Set the paths in a local copy of `configs/smoke_fastwam_joint.json` to the
matching LIBERO Joint checkpoint (`step_040000.pt`), its `dataset_stats.json`,
the Wan 2.2 TI2V 5B `Wan2.2_VAE.pth` and the directory containing the task's
cached T5 embedding. Paths in the JSON file are relative to that file. An
explicit `--checkpoint-path` is relative to the current directory. The example
below uses a copy of the configuration at `outputs/fastwam_joint/assets.json`.

Run the fixture exporter once **in the original Qi/LIBERO environment**. Give
it the source checkout and the local directory containing
`Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`. Then return to the project environment
for all offline inference and benchmarking.

```bash
python tools/prepare_fastwam_joint_reference.py \
  --reference-root /path/to/fastwamjoint \
  --wan-base /path/to/physical_WM/checkpoints \
  --output-dir outputs/fastwam_joint/smoke
```

The input NPZ contains the raw uint8 `agentview_image` and
`robot0_eye_in_hand_image`, `robot0_eef_pos`, `robot0_eef_quat`,
`robot0_gripper_qpos` and the unformatted task description in the Unicode
scalar `task_description` (also accepted as `prompt`). The service applies the native camera rotation, state
normalization, action denormalization and gripper conversion. For a paired
reference, the exporter saves the original evaluator's dense and K56 eager actions for the
same initial state, observation and seed in `reference_actions.npz` under
`dense_action` and `sparse_action`. Each array has shape `[32, 7]` in LIBERO
units. The reference benchmark uses `libero_spatial` task 0, initial state 0,
five dummy environment steps and seed 42.

```bash
CHECKPOINT=/path/to/checkpoints/step_040000.pt
ASSETS=outputs/fastwam_joint/assets.json
FIXTURE=outputs/fastwam_joint/smoke
python examples/infer_fastwam_joint.py \
  --checkpoint-path "$CHECKPOINT" --native-config "$ASSETS" \
  --observation "$FIXTURE/observation.npz" --mode sparse --execution eager \
  --output outputs/fastwam_joint/action.npz
```

## Paired 4090 validation

Run a one-warmup, one-measured-pair smoke first. The reference comparison
reports maximum absolute error and relative L2 separately for Dense and K56.
To enforce a numerical tolerance established on the original runtime, pass
`--reference-atol` in the selected output units. The exporter also records
preprocessed inputs, the top six blocks and per-frame Core/shared Stable
positions. The verifier checks those outside the timed region.

```bash
python tools/benchmark_policy.py --backbone fastwam-joint --smoke \
  --execution compile-graph --checkpoint-path "$CHECKPOINT" --native-config "$ASSETS" \
  --observation "$FIXTURE/observation.npz" --reference-actions "$FIXTURE/reference_actions.npz" \
  --seed 42 --output outputs/fastwam_joint/smoke-compile-graph.json

for backend in eager compile compile-graph; do
  python tools/benchmark_policy.py --backbone fastwam-joint \
    --execution "$backend" --checkpoint-path "$CHECKPOINT" --native-config "$ASSETS" \
    --observation "$FIXTURE/observation.npz" --seed 42 --warmup 2 --repeats 20 \
    --output "outputs/fastwam_joint/paired-$backend.json"
done
```

Each pair runs on one loaded model with the same observation and initial noise.
The service latency includes preprocessing, inference and conversion to LIBERO
actions. Run the same benchmark with `--timing-scope model` and a separate
output filename to time inference on prepared device inputs and compare
normalized model actions. In that mode `--reference-actions` selects the
exporter's `dense_normalized` and `sparse_normalized` arrays.
Reports record warmup samples (including the first
compile/capture), steady median and P90, the dense/sparse median ratio, peak
CUDA memory and per-chunk graph replay deltas.

| Check | Required result |
|---|---|
| Action | Finite `[32, 7]` |
| Step 0 | 30 blocks, 326 tokens, fused attention profile |
| Steps 1–9 | 144 queries, 242 keys/values, 19 frame-specific + 37 shared tokens per future latent |
| State | One initial pack, Step0 cross-attention KV reused, nine background updates |
| Compile | Entire sparse block stack compiled with no fallback |
| Compile + Graph | Capture passes native error limits, one capture for the fixed signature, nine replays per measured sparse chunk |

Single-observation latency and action comparisons cover offline inference.
Task success rates require a separate closed-loop LIBERO evaluation.

## RTX 4090 single-chunk results

Measured on 2026-11-26 with Torch 2.14.1+cu130, Triton 3.8.0 and the Conda
GCC 13 toolchain. Each row uses the recorded task 0 observation above, two
warmup pairs, and 20 alternating measured pairs on one loaded BF16 model.
The policy timings include LIBERO preprocessing and action conversion.

| Sparse execution | Dense eager median | Sparse median | Sparse P90 | Speedup |
|---|---:|---:|---:|---:|
| Eager | 438.27 ms | 314.07 ms | 366.68 ms | 1.40× |
| Compile | 433.46 ms | 229.58 ms | 233.63 ms | 1.89× |
| Compile + CUDA Graph | 433.50 ms | 219.84 ms | 224.48 ms | 1.97× |

The first sparse call with an empty Inductor cache took 51.79 s including
compilation and graph capture. The graph replayed nine times per measured
chunk, with a single capture and zero recorded compile or graph fallbacks.
Peak allocated GPU memory in the graph benchmark was 12.89 GiB. With prepared
model inputs, dense and K56 compile-graph medians were 431.83 and 218.05 ms
(1.98×).

The reference used the original working tree with Torch 2.7.1+cu128. On that
stack the port produced identical normalized and LIBERO actions for both eager
paths. On the locked stack the input tensors, selected blocks and both token
masks matched the reference. Maximum absolute differences in LIBERO action
units were 0.00732 for dense eager and 0.01099 for K56 eager on this fixture.
