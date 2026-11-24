"""Reports must identify real source/input content without requiring a Git checkout."""

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from sparse_wam.reproducibility import collect_run_metadata, source_metadata


def test_archive_records_content_hashes_and_unknown_commit(tmp_path):
    code = tmp_path / "src/sparse_wam/runner.py"
    code.parent.mkdir(parents=True)
    code.write_text("original = True\n")
    manifest = tmp_path / "SOURCE_MANIFEST.json"
    manifest.write_text("{}\n")
    native = tmp_path / "native.json"
    native.write_text('{"decode_video": false}\n')
    observation = tmp_path / "input.npz"
    observation.write_bytes(b"recorded-input")
    metadata = collect_run_metadata(
        root=tmp_path, argv=["benchmark", "--seed", "0"], native_config=native,
        input_files={"observation": observation},
    )
    assert metadata["argv"] == ["benchmark", "--seed", "0"]
    assert metadata["source"]["git_commit"] is None
    assert metadata["source"]["git_dirty"] is None
    assert metadata["source"]["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert metadata["native_config"]["sha256"] == hashlib.sha256(native.read_bytes()).hexdigest()
    assert metadata["input_files"]["observation"]["sha256"] == hashlib.sha256(observation.read_bytes()).hexdigest()
    previous_hash = metadata["source"]["implementation_sha256"]
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/report.json").write_text("{}")
    assert source_metadata(tmp_path)["implementation_sha256"] == previous_hash
    code.write_text("original = False\n")
    assert source_metadata(tmp_path)["implementation_sha256"] != previous_hash


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is optional for source archives")
def test_git_identifies_dirty_tree_without_attributing_parent_repository(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "--allow-empty", "-m", "test fixture")
    clean = source_metadata(tmp_path)
    assert clean["git_commit"] == git("rev-parse", "HEAD")
    assert clean["git_dirty"] is False
    (tmp_path / "new.py").write_text("changed = True\n")
    dirty = source_metadata(tmp_path)
    assert dirty["git_dirty"] is True
    assert "new.py" in dirty["git_status"]
    archive = tmp_path / "extracted_archive"
    archive.mkdir()
    assert source_metadata(archive)["git_commit"] is None


def test_missing_optional_files_are_unknown(tmp_path):
    result = collect_run_metadata(root=tmp_path, argv=[], native_config=tmp_path / "absent.json")
    assert result["native_config"]["sha256"] is None
    assert result["source"]["manifest_sha256"] is None
    assert result["source"]["implementation_sha256"] is None
    assert result["input_files"] == {}


def test_unavailable_git_does_not_prevent_reports(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(subprocess, "run", missing_git)
    result = source_metadata(Path(tmp_path))
    assert result["git_commit"] is None
    assert result["git_dirty"] is None
