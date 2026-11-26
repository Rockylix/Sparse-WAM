"""Paired Dense/Sparse inference on one recorded DROID or LIBERO observation.

The input NPZ stores numeric arrays and NumPy Unicode scalars. Policy timing
includes preprocessing, generation and action postprocessing. Joint model
timing uses prepared device inputs. Both scopes exclude initial model load.
"""

import argparse
import copy
from contextlib import nullcontext
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

from sparse_wam.config import BACKBONES, SparseWAMConfig
from sparse_wam.benchmark_execution import PairedExecution, ReplayCounter
from sparse_wam.reproducibility import collect_run_metadata, file_sha256
from sparse_wam.service import create_service


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=BACKBONES, required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--native-config")
    p.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    p.add_argument("--dense-eager", action="store_true", help="Keep Dense eager even when Sparse is compiled")
    p.add_argument("--no-nano-packing-cache", action="store_true")
    p.add_argument("--seed", type=int, help="Default: 42 for Joint, 0 for Cosmos")
    p.add_argument("--timing-scope", choices=("policy", "model"), default="policy",
                   help="Use model to time Joint inference on preprocessed inputs")
    p.add_argument("--reference-actions", type=Path,
                   help="Joint reference NPZ from tools/prepare_fastwam_joint_reference.py")
    p.add_argument("--reference-atol", type=float,
                   help="Fail if any absolute reference error exceeds this tolerance in the selected units")
    p.add_argument("--smoke", action="store_true", help="Validate one warmup and one measured pair")
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = p.parse_args(argv)
    joint = args.backbone == "fastwam-joint"
    if args.seed is None:
        args.seed = 42 if joint else 0
    if args.timing_scope == "model" and not joint:
        p.error("--timing-scope model requires --backbone fastwam-joint")
    if args.reference_actions and not joint:
        p.error("--reference-actions requires FastWAM-Joint")
    if args.reference_atol is not None and (
        not args.reference_actions or not np.isfinite(args.reference_atol) or args.reference_atol < 0
    ):
        p.error("--reference-atol requires --reference-actions and a finite, nonnegative tolerance")
    args.warmup = args.warmup if args.warmup is not None else (1 if args.smoke else 5)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.smoke else 30)
    if args.warmup < 1 or args.repeats < 1:
        p.error("warmup and repeats must be positive")
    if args.smoke and not joint and (args.backbone != "edge" or args.execution != "eager"):
        p.error("--smoke accepts Edge eager and FastWAM-Joint")
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


def validate_joint_result(mode, action, summary, execution, replay_delta, warmed):
    """Check actual execution, including the model's manual graph counter."""
    if action.shape != (32, 7) or not np.isfinite(action).all():
        raise RuntimeError(f"Joint {mode}: expected finite action (32, 7), got {action.shape}")
    sparse = mode == "sparse"
    graph = sparse and execution == "compile-graph"
    expected = {
        "backbone": "fastwam-joint", "mode": mode, "sparse_enabled": sparse,
        "execution": execution if sparse else "eager",
        "dense_forwards": 1 if sparse else 10, "sparse_forwards": 9 if sparse else 0,
    }
    if sparse:
        expected.update(
            dense_steps_executed=[0], completed_steps=list(range(10)), sparse_steps=list(range(1, 10)),
            profile_blocks=list(range(30)), dense_total_tokens=326, fixed_tokens_per_future=56,
            core_tokens_per_future=19, stable_tokens_shared=37, condition_kv_cache_tokens=98,
            sparse_query_tokens=144, sparse_key_value_tokens=242, single_pack_per_request=True,
            background_cache_source_step=0, background_cache_update_steps=list(range(1, 10)),
            cross_attention_kv_reused_from_step0=True, fused_step0_profile_used=True,
            fused_step0_profile_error=None, torch_compile_requested=execution != "eager",
            torch_compile_fallback_error=None, cuda_graph_requested=graph, cuda_graph_fallback_error=None,
        )
        if execution != "eager":
            expected["torch_compile_boundary"] = "step1_N_complete_B0_B29_sparse_stack"
    for key, value in expected.items():
        if key not in summary or summary[key] != value:
            raise RuntimeError(f"Joint {mode}: {key}={summary.get(key)!r}, expected {value!r}")
    if graph:
        validation = summary.get("cuda_graph_capture_validation") or {}
        for key, limit in (("relative_l2", 5e-3), ("max_absolute_error", 5e-2)):
            error = validation.get(key, float("inf"))
            if not np.isfinite(error) or not (0 <= error <= limit):
                raise RuntimeError(f"Joint sparse: invalid graph capture {key}={error!r}")
        if summary.get("cuda_graph_capture_count") != 1:
            raise RuntimeError("Joint sparse: expected one CUDA graph capture for a fixed input")
    if (warmed or not graph) and replay_delta != (9 if graph else 0):
        raise RuntimeError(f"Joint {mode}: unexpected CUDA graph replay delta {replay_delta}")


def read_reference_actions(path, model_only=False):
    suffix = "normalized" if model_only else "action"
    with np.load(path, allow_pickle=False) as archive:
        actions = {mode: archive[f"{mode}_{suffix}"].astype(np.float64) for mode in ("dense", "sparse")}
    for mode, action in actions.items():
        if action.shape != (32, 7) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite [32, 7] {mode} reference in {path}")
    return actions


def compare_reference(action, reference, mode, atol):
    difference = action.astype(np.float64) - reference
    max_abs = float(np.abs(difference).max())
    if atol is not None and max_abs > atol:
        raise RuntimeError(f"Joint {mode}: reference max absolute error {max_abs} exceeds {atol}")
    return {"max_absolute_error": max_abs,
            "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12))}


def check_reference_fixture(path, service, execution, observation, *, prepared=None):
    """Check optional native input and selection goldens outside timed calls."""
    checked = []
    with np.load(path, allow_pickle=False) as archive:
        selection = {"selected_blocks", "core_masks", "stable_mask"}.intersection(archive.files)
        if not selection and "image" not in archive and "proprio" not in archive:
            return checked
        if prepared is None:
            prepared = service.prepare_observation(observation)
        for name, argument in (("image", "input_image"), ("proprio", "proprio")):
            if name in archive:
                actual = prepared[argument].detach().float().cpu().numpy()
                np.testing.assert_allclose(actual, archive[name], rtol=1e-6, atol=1e-6, err_msg=name)
                checked.append(name)
        if selection:
            execution.select("sparse")
            runner = service.sparse_wam_runner
            runner.capture_selection = True
            try:
                service.infer_prepared(prepared)
            finally:
                runner.capture_selection = False
            summary = runner.last_summary
            if "selected_blocks" in selection:
                np.testing.assert_array_equal(summary["selected_blocks_ranked"], archive["selected_blocks"],
                                              err_msg="selected_blocks")
                checked.append("selected_blocks")
            if "core_masks" in selection:
                positions = [np.flatnonzero(row) for row in archive["core_masks"]]
                np.testing.assert_array_equal(summary["core_positions"], positions, err_msg="core_positions")
                checked.append("core_positions")
            if "stable_mask" in selection:
                np.testing.assert_array_equal(summary["stable_positions"], np.flatnonzero(archive["stable_mask"]),
                                              err_msg="stable_positions")
                checked.append("stable_positions")
    return checked


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
    model_only = args.timing_scope == "model"
    references = read_reference_actions(args.reference_actions, model_only) if args.reference_actions else None
    reference_info = provenance["observation"] or {}
    if references and reference_info.get("seed", args.seed) != args.seed:
        raise ValueError("Benchmark seed differs from the reference observation seed")
    if references and reference_info.get("reference_actions_sha256"):
        if file_sha256(args.reference_actions) != reference_info["reference_actions_sha256"]:
            raise ValueError("Reference actions do not match the observation provenance")
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
    joint = args.backbone == "fastwam-joint"
    sample = service.prepare_observation(observation) if model_only else observation
    timings = {"dense": [], "sparse": []}
    warmup_timings = {"dense": [], "sparse": []}
    reference_errors = {"dense": [], "sparse": []}
    last_joint_replays = 0
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
            if not joint:
                service._rng = np.random.default_rng(args.seed + round_id)
                torch.manual_seed(args.seed + round_id)
                torch.cuda.manual_seed_all(args.seed + round_id)
                sample = copy.deepcopy(observation)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with (nullcontext() if joint else ReplayCounter()) as counter:
                result = service.infer_prepared(sample) if model_only else service.infer(sample)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            action = result["action"]
            if isinstance(action, torch.Tensor):
                action = action.detach().float().cpu().numpy()
            outputs[mode] = np.asarray(action)
            summaries[mode] = {**runner.last_summary, "checkpoint": args.checkpoint_path}
            if joint:
                cumulative = summaries[mode]["cuda_graph_replay_count"] if mode == "sparse" else last_joint_replays
                replay_delta = cumulative - last_joint_replays
                last_joint_replays = cumulative
                validate_joint_result(mode, outputs[mode], summaries[mode], args.execution,
                                      replay_delta, warmed=round_id >= args.warmup)
                if references:
                    reference_errors[mode].append(compare_reference(
                        outputs[mode], references[mode], mode, args.reference_atol,
                    ))
            else:
                replay_delta = counter.count
            if round_id >= args.warmup:
                timings[mode].append(elapsed)
                replays[mode].append(replay_delta)
            else:
                warmup_timings[mode].append(elapsed)
            action_shapes[mode] = list(outputs[mode].shape)
            if args.smoke:
                if not joint:
                    validate_smoke_result(mode, result, summaries[mode])
                smoke_checks.append({"round": round_id, "mode": mode, "passed": True})
            print(
                f"round={round_id} warmup={round_id < args.warmup} mode={mode} "
                f"elapsed={elapsed:.6f}s graph_replays={replay_delta}",
                flush=True,
            )
        if round_id >= args.warmup:
            delta = outputs["sparse"] - outputs["dense"]
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
        "warmup_samples_s": warmup_timings,
        "cuda_graph_replays_by_round": replays,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "timing_scope": ("prepared Joint inputs: infer_action with controller setup and summary"
                         if model_only else "policy.infer: preprocessing + generation + action postprocessing"),
        "stats": stats,
        "speedup": stats["dense"]["median_s"] / stats["sparse"]["median_s"],
        "action_mse_by_pair": action_error,
        "effective_runs": summaries,
        "action_shapes": action_shapes,
        "environment": environment_versions(),
        "provenance": provenance,
        "reproduction": reproduction,
        "note": ("Joint dense is native eager for every sparse execution backend." if joint else
                 "Nano dense is native eager even when its sparse adapter is compiled; no task success claim."),
    }
    if references:
        report["reference_actions"] = {
            "path": str(args.reference_actions), "absolute_tolerance": args.reference_atol,
            "errors_by_round": reference_errors,
            "units": "normalized model output" if model_only else "LIBERO",
            "fixture_checks": check_reference_fixture(args.reference_actions, service, execution, observation,
                                                      prepared=sample if model_only else None),
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
        inputs = {"observation": args.observation}
        if args.reference_actions:
            inputs["reference_actions"] = args.reference_actions
        if args.backbone == "fastwam-joint":
            inputs["checkpoint"] = args.checkpoint_path
        reproduction = collect_run_metadata(
            argv=sys.argv if argv is None else [sys.argv[0], *argv],
            native_config=args.native_config, input_files=inputs,
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
