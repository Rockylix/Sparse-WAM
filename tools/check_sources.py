"""Verify the source provenance and detect unexpected runtime edits, without Git/network."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    modified = {entry["path"]: entry for entry in manifest["local_changes"]}
    failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        expected = modified.get(entry["path"], {}).get("sha256", entry["source_sha256"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(entry["path"])
    if failures:
        raise SystemExit("Source mismatch: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} source files and {len(modified)} documented local changes")


if __name__ == "__main__":
    main()
