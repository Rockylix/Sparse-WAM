"""Paired Dense/Sparse RoboLab observation benchmark. No robot connection or movement.

Input NPZ contains the same keys as an openpi observation; no pickle is accepted.
Strings such as `prompt` are NumPy Unicode scalars. Measurements include native
preprocessing, generation, and action postprocessing, excluding initial model load.
"""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from sparse_wam.config import SparseWAMConfig
from sparse_wam.benchmark_execution import PairedExecution, ReplayCounter
from sparse_wam.service import create_service


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=("edge", "nano"), required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--native-config")
    p.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    p.add_argument("--dense-eager", action="store_true", help="Keep Dense eager even when Sparse is compiled")
    p.add_argument("--no-nano-packing-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        p.error("warmup and repeats must be positive")
    with np.load(args.observation, allow_pickle=False) as source:
        observation = {
            key: source[key].item() if source[key].ndim == 0 else source[key].copy() for key in source.files
        }
    config = SparseWAMConfig(args.backbone, args.execution, nano_packing_cache=not args.no_nano_packing_cache)
    service, _ = create_service(
        config,
        checkpoint=args.checkpoint_path,
        seed=args.seed,
        native_config=args.native_config,
        mode="dense",
    )
    runner = service.sparse_wam_runner
    execution = PairedExecution(service, config, dense_eager=args.dense_eager)
    timings = {"dense": [], "sparse": []}
    replays = {"dense": [], "sparse": []}
    action_error = []
    summaries = {}
    for round_id in range(args.warmup + args.repeats):
        outputs = {}
        for mode in ("dense", "sparse") if round_id % 2 == 0 else ("sparse", "dense"):
            execution.select(mode)
            # Same observation and initial generation seed for each pair.
            service._rng = np.random.default_rng(args.seed + round_id)
            torch.manual_seed(args.seed + round_id)
            torch.cuda.manual_seed_all(args.seed + round_id)
            sample = copy.deepcopy(observation)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with ReplayCounter() as counter:
                outputs[mode] = service.infer(sample)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if round_id >= args.warmup:
                timings[mode].append(elapsed)
                replays[mode].append(counter.count)
            summaries[mode] = runner.last_summary
            print(
                f"round={round_id} warmup={round_id < args.warmup} mode={mode} "
                f"elapsed={elapsed:.6f}s graph_replays={counter.count}",
                flush=True,
            )
        if round_id >= args.warmup:
            delta = outputs["sparse"]["action"] - outputs["dense"]["action"]
            action_error.append(float(np.mean(delta**2)))
    stats = {
        mode: {
            "mean_s": float(np.mean(values)),
            "median_s": float(np.median(values)),
            "p90_s": float(np.percentile(values, 90)),
            "samples_s": values,
        }
        for mode, values in timings.items()
    }
    report = {
        "config": config.to_dict(),
        "checkpoint": args.checkpoint_path,
        "observation": str(args.observation),
        "seed": args.seed,
        "warmup_rounds": args.warmup,
        "measured_rounds": args.repeats,
        "dense_eager_requested": args.dense_eager,
        "cuda_graph_replays_by_round": replays,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "timing_scope": "policy.infer: preprocessing + generation + action postprocessing",
        "stats": stats,
        "speedup": stats["dense"]["median_s"] / stats["sparse"]["median_s"],
        "action_mse_by_pair": action_error,
        "effective_runs": summaries,
        "note": "Nano dense is native eager even when its sparse adapter is compiled; no task success claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
