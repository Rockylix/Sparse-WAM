"""Small, model-free provenance records shared by offline experiment tools."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from . import __version__


def file_sha256(path):
    """Hash an existing file; absent optional provenance is recorded as unknown."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _git(root, *args):
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=5,
        ).stdout.rstrip("\n")
    except (OSError, subprocess.SubprocessError):
        return None


def source_metadata(root=None):
    """Describe this source tree, without borrowing an enclosing repository's SHA.

    The implementation hash covers Python source/tools/examples and JSON configurations,
    plus the environment specification. Vendored sources have a separate manifest
    digest and are checked by tools/check_sources.py.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    commit = status = None
    if (root / ".git").exists():
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    files = {}
    for pattern in ("src/sparse_wam/**/*.py", "tools/**/*.py", "examples/**/*.py", "configs/**/*.json"):
        for path in sorted(root.glob(pattern)):
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        if (root / name).is_file():
            files[name] = file_sha256(root / name)
    return {
        "version": __version__,
        "git_commit": commit,
        "git_dirty": status != "" if status is not None else None,
        "git_status": status,
        "manifest_sha256": file_sha256(root / "SOURCE_MANIFEST.json"),
        "implementation_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if files else None,
    }


def collect_run_metadata(*, argv=None, native_config=None, input_files=None, root=None):
    """Capture invocation and file hashes before timing; never serialize environment secrets.

    Paths and argv are local reproduction information. Review/redact them before
    publishing a report. Missing files get a null hash; input validation remains
    the caller's responsibility.
    """
    def record(path):
        return {"path": str(path), "sha256": file_sha256(path)}

    return {
        "source": source_metadata(root),
        "argv": list(sys.argv if argv is None else argv),
        "native_config": record(native_config) if native_config is not None else None,
        "input_files": {name: record(path) for name, path in (input_files or {}).items()},
    }
