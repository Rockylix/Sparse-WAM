"""Export pinned runtime snapshots without Git history, weights, or experiment artifacts.

Run once from a fresh destination. Source checkouts are read-only inputs.
The two framework ABIs intentionally remain separate until GPU migration validation.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "edge": (
        "Cosmos3-Task-relevant-Inference",
        "version/core80-stable104-action-weighted",
        "deb4af65000d06b7b13e30c6ebcc0d44e6c67921",
    ),
    "nano": ("Cosmos3-nano-policy", "version/asi", "a89509e1f54e60f0622741995efd5e2f3d6710d6"),
}


def main():
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
