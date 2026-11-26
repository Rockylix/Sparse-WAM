"""Export pinned runtime snapshots without Git history, weights, or experiment artifacts.

Run once from a fresh destination. Source checkouts are read-only inputs.
The Edge and Nano snapshots have separate framework ABIs. The Joint export
uses its pinned working-tree files and the recorded packaging patch.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "edge": (
        "Cosmos3-Task-relevant-Inference",
        "version/core80-stable104-action-weighted",
        "deb4af65000d06b7b13e30c6ebcc0d44e6c67921",
    ),
    "nano": ("Cosmos3-nano-policy", "version/asi", "a89509e1f54e60f0622741995efd5e2f3d6710d6"),
}


def export_fastwam_joint(source, destination):
    """Reproduce the published BF16 snapshot from the recorded K56 working tree."""
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    source_info = manifest["sources"]["fastwam_joint"]
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != source_info["commit"]:
        raise ValueError(f"Expected FastWAM-Joint base commit {source_info['commit']}, found {head}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")

    entries = [entry for entry in manifest["files"] if entry["source"] == "fastwam_joint"]
    changes = {entry["path"]: entry["sha256"] for entry in manifest["local_changes"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        for entry in entries:
            data = (source / entry["source_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["source_sha256"]:
                raise ValueError(f"Reference working-tree mismatch: {entry['source_path']}")
            output = staged / Path(entry["path"]).relative_to("vendor/fastwam_joint")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(ROOT / "tools/fastwam_joint_runtime.patch")],
            cwd=staged, check=True,
        )
        for entry in entries:
            output = staged / Path(entry["path"]).relative_to("vendor/fastwam_joint")
            expected = changes.get(entry["path"], entry["source_sha256"])
            if hashlib.sha256(output.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Adapted file mismatch: {entry['path']}")
        staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastwam-joint-source", type=Path, help="Export only the pinned Joint working tree")
    parser.add_argument("--destination", type=Path, help="Output directory for a fresh Joint export")
    args = parser.parse_args()
    if args.fastwam_joint_source:
        export_fastwam_joint(args.fastwam_joint_source, args.destination or ROOT / "vendor/fastwam_joint")
        return
    if args.destination:
        parser.error("--destination requires --fastwam-joint-source")
    manifest = {"sources": {}, "files": [], "local_changes": []}
    for tier, (name, branch, commit) in SOURCES.items():
        source = ROOT.parent / name
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError(f"{tier}: expected {commit}, got {actual}")
        destination = ROOT / "vendor" / tier
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing runtime: {destination}")
        manifest["sources"][tier] = {"repository": f"Rockylix/{name}", "branch": branch, "commit": commit}
        tracked = subprocess.check_output(["git", "-C", str(source), "ls-files"], text=True).splitlines()
        for filename in tracked:
            path = Path(filename)
            include = filename.startswith(("cosmos_framework/", "packages/")) or filename in (
                "pyproject.toml",
                "uv.lock",
                "LICENSE",
                ".python-version",
                "MANIFEST.in",
            )
            if not include or path.name.endswith(("_test.py",)) or path.name.startswith("test_"):
                continue
            if path.suffix in (".pt", ".pth", ".safetensors"):
                continue
            data = (source / filename).read_bytes()
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest["files"].append(
                {
                    "path": target.relative_to(ROOT).as_posix(),
                    "source": tier,
                    "source_path": filename,
                    "source_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    shutil.copyfile(ROOT / "vendor/edge/LICENSE", ROOT / "LICENSE")
    # The model-independent layout resolver is copied intact and remains auditable.
    filename = "cosmos_framework/inference/edge_core_stable_layout.py"
    target = ROOT / "src/sparse_wam/layout.py"
    data = (ROOT / "vendor/edge" / filename).read_bytes()
    target.write_bytes(data)
    manifest["files"].append(
        {
            "path": target.relative_to(ROOT).as_posix(),
            "source": "edge",
            "source_path": filename,
            "source_sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    (ROOT / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest['files'])} source files; no .git directories or remotes")


if __name__ == "__main__":
    main()
