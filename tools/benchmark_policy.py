"""Paired Dense/Sparse RoboLab observation benchmark. No robot connection or movement.

Input NPZ contains the same keys as an openpi observation; no pickle is accepted.
Strings such as `prompt` are NumPy Unicode scalars. Measurements include native
preprocessing, generation, and action postprocessing, excluding initial model load.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from sparse_wam.config import SparseWAMConfig
from sparse_wam.benchmark_execution import PairedExecution, ReplayCounter
from sparse_wam.reproducibility import collect_run_metadata
from sparse_wam.service import create_service


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=("edge", "nano"), required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--native-config")
    p.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    p.add_argument("--dense-eager", action="store_true", help="Keep Dense eager even when Sparse is compiled")
    p.add_argument("--no-nano-packing-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="Validate Edge eager Dense/Sparse with one warmup and one measured pair")
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = p.parse_args(argv)
    args.warmup = args.warmup if args.warmup is not None else (1 if args.smoke else 5)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.smoke else 30)
    if args.warmup < 1 or args.repeats < 1:
        p.error("warmup and repeats must be positive")
    if args.smoke and (args.backbone != "edge" or args.execution != "eager"):
        p.error("--smoke requires --backbone edge --execution eager")
    if args.smoke and (args.warmup != 1 or args.repeats != 1):
        p.error("--smoke requires --warmup 1 --repeats 1")
    return args


def validate_smoke_result(mode, output, summary):
    """Reject superficially successful runs that skipped the expected sparse work."""
    action = np.asarray(output["action"])
    if action.shape != (32, 8) or not np.isfinite(action).all():
        raise RuntimeError(f"Smoke {mode}: expected finite action (32, 8), got {action.shape}")
    expected = {
        "backbone": "edge", "execution": "eager", "mode": mode,
        "sparse_enabled": mode == "sparse", "spatial_tokens": 340,
        "dense_forwards": 8 if mode == "dense" else 1,
        "sparse_forwards": 0 if mode == "dense" else 7,
        "retained_tokens": 340 if mode == "dense" else 184,
    }
    if mode == "sparse":
        expected.update(
            completed_forwards=8, block_calls=28 * 8, pack_count=7, restore_count=7,
            velocity_updates=4, dense_gen_tokens=3093, sparse_gen_tokens=1845,
        )
    if not isinstance(summary, dict):
        raise RuntimeError(f"Smoke {mode}: missing runner summary")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Smoke {mode}: {key}={summary.get(key)!r}, expected {value!r}")


def collect_provenance(checkpoint_path, observation):
    checkpoint = Path(checkpoint_path)
    result = {}
    for name, sidecar in (
        ("checkpoint", checkpoint / "provenance.json"),
        ("observation", observation.with_suffix(".provenance.json")),
    ):
        result[name] = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
    result["checkpoint_revision"] = (result["checkpoint"] or {}).get("resolved_revision")
    if result["checkpoint_revision"] is None:
        resolved = checkpoint.resolve()
        if resolved.parent.name == "snapshots":
            result["checkpoint_revision"] = resolved.name
    expected_hash = (result["observation"] or {}).get("observation_sha256")
    if expected_hash is not None:
        with observation.open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Observation SHA256 does not match provenance: {observation}")
    return result


def environment_versions():
    packages = {}
    for name in ("torchvision", "diffusers", "transformers", "transformer-engine", "nvidia-cudnn-cu13"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "architecture": platform.machine(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "packages": packages,
    }


def run_benchmark(args, started_at, reproduction=None):
    provenance = collect_provenance(args.checkpoint_path, args.observation)
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
    action_shapes = {}
    smoke_checks = []
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
            summaries[mode] = {**runner.last_summary, "checkpoint": args.checkpoint_path}
            action_shapes[mode] = list(np.asarray(outputs[mode]["action"]).shape)
            if args.smoke:
                validate_smoke_result(mode, outputs[mode], summaries[mode])
                smoke_checks.append({"round": round_id, "mode": mode, "passed": True})
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
        "started_at_utc": started_at,
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
        "action_shapes": action_shapes,
        "environment": environment_versions(),
        "provenance": provenance,
        "reproduction": reproduction,
        "note": "Nano dense is native eager even when its sparse adapter is compiled; no task success claim.",
    }
    if args.smoke:
        report["smoke"] = {"passed": True, "status": "passed", "checks": smoke_checks}
    write_report(args.output, report)
    print(json.dumps(stats, indent=2))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    if args.smoke:
        write_report(args.output, {
            "started_at_utc": started_at, "smoke": {"passed": False, "status": "running"},
        })
    reproduction = None
    try:
        reproduction = collect_run_metadata(
            argv=sys.argv if argv is None else [sys.argv[0], *argv],
            native_config=args.native_config, input_files={"observation": args.observation},
        )
        run_benchmark(args, started_at, reproduction)
    except Exception as exc:
        if args.smoke:
            write_report(args.output, {
                "started_at_utc": started_at,
                "reproduction": reproduction,
                "smoke": {"passed": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
            })
        raise


if __name__ == "__main__":
    main()
