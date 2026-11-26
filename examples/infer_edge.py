"""Offline Edge inference on a recorded NPZ observation. No robot connection.

Run from a complete editable source checkout, with the Edge dependencies and
local model assets prepared as described in README.md.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from sparse_wam import SparseWAMConfig
from sparse_wam.reproducibility import collect_run_metadata
from sparse_wam.service import create_service


def main(argv=None, *, backbone="edge"):
    parser = argparse.ArgumentParser(description=f"Run the {backbone} policy on a recorded observation")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--mode", choices=("dense", "sparse"), default="sparse")
    parser.add_argument("--execution", choices=("eager", "compile", "compile-graph"), default="eager")
    parser.add_argument("--seed", type=int, default=42 if backbone == "fastwam-joint" else 0)
    parser.add_argument("--output", type=Path, default=Path("outputs/example") / f"{backbone}_action.npz")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz; a .json summary is written alongside it")
    summary_path = args.output.with_suffix(".json")
    if args.output.exists() or summary_path.exists():
        parser.error("output or summary already exists; choose a new --output")
    with np.load(args.observation, allow_pickle=False) as source:
        observation = {
            key: source[key].item() if source[key].ndim == 0 else source[key].copy() for key in source.files
        }
    reproduction = collect_run_metadata(
        argv=sys.argv if argv is None else [sys.argv[0], *argv],
        native_config=args.native_config, input_files={"observation": args.observation},
    )
    service, settings = create_service(
        SparseWAMConfig(backbone, args.execution), checkpoint=args.checkpoint_path,
        native_config=args.native_config, seed=args.seed, mode=args.mode,
        output_dir=str(args.output.parent / "native"),
    )
    result = service.infer(observation)
    action = np.asarray(result["action"])
    if action.ndim != 2 or action.shape[0] != 32 or action.shape[1] == 0 or not np.isfinite(action).all():
        raise RuntimeError(f"Expected finite [32, action_dim] actions from the {backbone} policy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        np.savez(target, action=action)
    with summary_path.open("x", encoding="utf-8") as target:
        json.dump({
            "checkpoint": str(settings.checkpoint_path), "action_shape": list(action.shape),
            "summary": {**service.sparse_wam_runner.last_summary, "checkpoint": str(settings.checkpoint_path)},
            "reproduction": reproduction,
        }, target, indent=2)
        target.write("\n")
    print(f"Saved actions to {args.output} and run summary to {summary_path}")


if __name__ == "__main__":
    main()
