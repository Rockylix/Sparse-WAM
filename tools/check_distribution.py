"""Check release artifacts, then exercise the unpacked source distribution on CPU.

Run after ``uv build``. This does not load model weights, install dependencies,
request a GPU, or treat the wheel as a standalone Cosmos inference runtime.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import zipfile


REQUIRED_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "SOURCE_MANIFEST.json",
    "pyproject.toml", "uv.lock", ".python-version", "MANIFEST.in", "CONTRIBUTING.md",
    "src/sparse_wam/__init__.py", "src/sparse_wam/runtime.py",
    "tools/check_sources.py", "tools/check_distribution.py", "tools/prepare_smoke.py",
    "tools/benchmark_policy.py", "configs/edge.json", "configs/nano.json",
    "configs/smoke_edge.json", "docs/method.md", "docs/validation.md",
    "docs/api.md", "docs/rtx4090.md",
    "examples/infer_edge.py", ".github/workflows/ci.yml",
    "examples/infer_fastwam_joint.py", "configs/fastwam_joint.json",
    "configs/smoke_fastwam_joint.json", "docs/fastwam_joint.md",
    "src/sparse_wam/backends/fastwam_joint.py",
    "tools/prepare_fastwam_joint_reference.py",
)
SOURCE_DIRECTORIES = ("src/sparse_wam", "vendor", "tools", "configs", "tests", "docs", "examples")
FORBIDDEN_ROOTS = {"outputs", "checkpoints", "experiments", ".venv", "build", "dist"}
EXCLUDED_DOCUMENTATION = {"docs/evidence", "docs/changes.md"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".pt", ".pth", ".safetensors"}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def excluded_documentation(name):
    return any(name == path or name.startswith(f"{path}/") for path in EXCLUDED_DOCUMENTATION)


def validate_paths(names):
    for name in names:
        parts = PurePosixPath(name).parts
        require(parts and not name.startswith("/") and ".." not in parts, f"Unsafe archive path: {name}")
        require(parts[0] not in FORBIDDEN_ROOTS, f"Local artifacts in distribution: {name}")
        require(not excluded_documentation(name), f"Archived local documentation in distribution: {name}")
        require(not any(p in {".git", "__pycache__"} or p.startswith(".venv") for p in parts),
                f"Local cache in distribution: {name}")
        require(not any(p == ".env" or p.startswith(".env.") for p in parts),
                f"Environment file in distribution: {name}")
        require(PurePosixPath(name).suffix not in EXCLUDED_SUFFIXES,
                f"Generated code, logs or model weights in distribution: {name}")


def check_sdist(archive_path, destination, source_root):
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        require(len(roots) == 1, "Expected one source distribution root directory")
        root_name = roots.pop()
        require(root_name not in {"", ".", ".."}, "Invalid source distribution root")
        files = {}
        for member in members:
            require(member.isdir() or member.isfile(), f"Unexpected archive entry: {member.name}")
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, f"Unsafe archive path: {member.name}")
            if member.isfile():
                files[str(path.relative_to(root_name))] = member
        validate_paths(files)
        for filename in REQUIRED_FILES:
            require(filename in files, f"Source distribution is missing {filename}")
        # Compare the working tree's release directories as well as the fixed
        # essentials, so newly added modules, fixtures and docs cannot be lost.
        for directory in SOURCE_DIRECTORIES:
            for source in (source_root / directory).rglob("*"):
                if not source.is_file() or "__pycache__" in source.parts or source.suffix in EXCLUDED_SUFFIXES:
                    continue
                relative = source.relative_to(source_root).as_posix()
                if excluded_documentation(relative):
                    continue
                require(relative in files, f"Source distribution omitted {relative}")
                with archive.extractfile(files[relative]) as bundled:
                    require(bundled.read() == source.read_bytes(), f"Stale source distribution file: {relative}")
        archive.extractall(destination, filter="data")
    unpacked = destination / root_name
    manifest = json.loads((unpacked / "SOURCE_MANIFEST.json").read_text())
    overrides = {entry["path"]: entry["sha256"] for entry in manifest["local_changes"]}
    for entry in manifest["files"]:
        filename = entry["path"]
        require(filename in files, f"Missing attributed source file: {filename}")
        expected = overrides.get(filename, entry["source_sha256"])
        require(hashlib.sha256((unpacked / filename).read_bytes()).hexdigest() == expected,
                f"Source provenance mismatch: {filename}")
    print(f"Source distribution: {len(files)} files; {len(manifest['files'])} source hashes verified", flush=True)
    return unpacked


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_paths(names)
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md", "vendor/fastwam_joint/LICENSE"):
            require(any(name.endswith(f".dist-info/licenses/{filename}") for name in names),
                    f"Wheel is missing license attribution: {filename}")
        require("sparse_wam/runtime.py" in names, "Wheel is missing the public runtime module")
        metadata = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
        require(metadata is not None, "Wheel is missing package metadata")
        text = archive.read(metadata).decode()
        require("Description-Content-Type: text/markdown" in text and "# Sparse-WAM" in text,
                "Wheel metadata is missing the README")
    print(f"Wheel: {len(names)} files; metadata and attribution verified", flush=True)


def check_unpacked_source(root):
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                   "CUDA_VISIBLE_DEVICES": ""}
    commands = (
        ("tools/check_sources.py",),
        ("-m", "pytest", "-q"),
        ("-m", "sparse_wam", "--help"),
        ("-m", "sparse_wam", "serve", "--help"),
        ("-m", "sparse_wam", "config", "--backbone", "edge"),
        ("-m", "sparse_wam", "config", "--backbone", "nano"),
        ("-m", "sparse_wam", "config", "--backbone", "fastwam-joint"),
        ("examples/infer_edge.py", "--help"),
        ("examples/infer_fastwam_joint.py", "--help"),
    )
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=root, env=environment, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    sources, wheels = sorted(args.dist.glob("*.tar.gz")), sorted(args.dist.glob("*.whl"))
    require(len(sources) == len(wheels) == 1, "Expected exactly one source archive and one wheel in --dist")
    check_wheel(wheels[0])
    with tempfile.TemporaryDirectory(prefix="sparse-wam-distribution-") as temporary:
        root = check_sdist(sources[0], Path(temporary), Path(__file__).resolve().parents[1])
        check_unpacked_source(root)
    print("Release artifact checks passed; model/GPU inference was not tested.")


if __name__ == "__main__":
    main()
