"""Same-loaded-model Nano packing on/off regression and synchronized timing.

Uses the existing RoboLab observation format; no robot connection or movement.
Dense remains eager without packing reuse. Both Sparse modes use compile+Graph.
"""

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from sparse_wam.benchmark_execution import ReplayCounter
from sparse_wam.config import SparseWAMConfig
from sparse_wam.controller import SparseWAMController
from sparse_wam.reproducibility import collect_run_metadata
from sparse_wam.service import create_service


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--observation", type=Path, required=True)
    p.add_argument("--native-config", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        p.error("positive warmup and repeats required")
    if args.output.exists():
        p.error("output already exists")
    reproduction = collect_run_metadata(
        native_config=args.native_config, input_files={"observation": args.observation},
    )
    with np.load(args.observation, allow_pickle=False) as data:
        obs = {k: data[k].item() if data[k].ndim == 0 else data[k].copy() for k in data.files}
    config = SparseWAMConfig("nano", "compile-graph")
    service, _ = create_service(config, checkpoint=args.checkpoint_path, native_config=args.native_config)
    runner = service.sparse_wam_runner
    generate = runner.generate
    capture = {}

    def timed_generate(*a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(*a, **kw)
        torch.cuda.synchronize()
        capture["generation_s"] = time.perf_counter() - start
        capture["result"] = result
        return result

    runner.generate = timed_generate
    original_finish = SparseWAMController.finish

    def finish(controller):
        capture["plan"] = controller.plan
        return original_finish(controller)

    SparseWAMController.finish = finish
    rows, checks = [], []
    prefill = [0]

    def pre_hook(module, a, kw):
        if kw.get("und_only", a[2] if len(a) > 2 else False):
            prefill[0] += 1

    handle = service.model.net.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        for round_id in range(args.warmup + args.repeats):
            modes = ("dense", "packing_off", "packing_on")
            if round_id % 2:
                modes = tuple(reversed(modes))
            outputs = {}
            # Two RNG/input variants exercise new-chunk cache state, not just
            # repeated replay of one result. Within each round inputs match.
            seed = round_id % 2
            for mode in modes:
                capture.clear()
                runner.mode = "dense" if mode == "dense" else "sparse"
                runner.config = replace(config, nano_packing_cache=mode == "packing_on")
                sample = copy.deepcopy(obs)
                if seed:
                    image_key = "observation/image"
                    sample[image_key] = np.clip(sample[image_key].astype(np.int16) + 1, 0, 255).astype(
                        np.uint8
                    )
                service._rng = np.random.default_rng(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with ReplayCounter() as graph:
                    returned = service.infer(sample)
                torch.cuda.synchronize()
                row = dict(
                    round=round_id,
                    seed=seed,
                    mode=mode,
                    measured=round_id >= args.warmup,
                    infer_s=time.perf_counter() - start,
                    generation_s=capture["generation_s"],
                    graph_replays=graph.count,
                    summary={**runner.last_summary, "checkpoint": args.checkpoint_path},
                    prefill_calls=prefill[0],
                )
                rows.append(row)
                if mode != "dense":
                    outputs[mode] = {
                        k: capture["result"][k][0].detach().cpu().clone() for k in ("action", "vision")
                    }
                    outputs[mode].update(
                        mask=capture["plan"].keep.cpu().clone(),
                        layers=capture["plan"].layers.cpu().clone(),
                        quality=capture["plan"].quality.cpu().clone(),
                        returned_action=torch.from_numpy(returned["action"].copy()),
                    )
                print(
                    f"round={round_id} {mode} generation={row['generation_s']:.6f}s "
                    f"infer={row['infer_s']:.6f}s graph={graph.count} "
                    f"packing={row['summary'].get('input_packing')}",
                    flush=True,
                )
            equal = {
                k: torch.equal(outputs["packing_off"][k], outputs["packing_on"][k])
                for k in outputs["packing_on"]
            }
            checks.append(dict(round=round_id, exact=equal))
            if not all(equal.values()):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(dict(status="regression_failed", checks=checks, rows=rows,
                                    reproduction=reproduction), indent=2)
                )
                raise RuntimeError(f"Packing changed output: {equal}")
            assert prefill[0] == 2, "Instruction KV cache unexpectedly missed"
        stats = {}
        for mode in ("dense", "packing_off", "packing_on"):
            stats[mode] = {}
            for scope in ("generation_s", "infer_s"):
                values = [r[scope] for r in rows if r["mode"] == mode and r["measured"]]
                stats[mode][scope] = dict(
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    p90=float(np.percentile(values, 90)),
                )
        report = dict(
            status="passed",
            stats=stats,
            checks=checks,
            rows=rows,
            prefill_calls=prefill[0],
            config=config.to_dict(),
            checkpoint=args.checkpoint_path,
            observation=str(args.observation),
            reproduction=reproduction,
            note="Diagnostic nested synchronization; same model, paired two input/seed variants; no task success claim",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(stats, indent=2), flush=True)
    finally:
        handle.remove()
        SparseWAMController.finish = original_finish
        runner.generate = generate
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
