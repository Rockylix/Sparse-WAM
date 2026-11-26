"""Smoke validation must reject plausible outputs with incomplete sparse execution."""

import importlib.util
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparse_wam.config import SparseWAMConfig


spec = importlib.util.spec_from_file_location(
    "benchmark_policy", Path(__file__).resolve().parents[1] / "tools" / "benchmark_policy.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def sparse_summary():
    return {
        **SparseWAMConfig("edge").run_metadata("sparse"),
        "completed_forwards": 8, "block_calls": 224, "pack_count": 7, "restore_count": 7,
        "velocity_updates": 4, "dense_gen_tokens": 3093, "sparse_gen_tokens": 1845,
    }


def cli_args(tmp_path):
    return ["--backbone", "edge", "--checkpoint-path", str(tmp_path / "checkpoint"),
            "--observation", str(tmp_path / "observation.npz"), "--output", str(tmp_path / "report.json")]


def test_smoke_defaults_and_regular_benchmark_defaults(tmp_path):
    regular = benchmark.parse_args(cli_args(tmp_path))
    smoke = benchmark.parse_args([*cli_args(tmp_path), "--smoke"])
    assert (regular.warmup, regular.repeats) == (5, 30)
    assert (smoke.warmup, smoke.repeats, smoke.execution) == (1, 1, "eager")


@pytest.mark.parametrize("options", [
    ["--backbone", "nano"], ["--execution", "compile"], ["--warmup", "2"], ["--repeats", "2"],
])
def test_smoke_rejects_other_execution_protocols(tmp_path, options):
    with pytest.raises(SystemExit) as error:
        benchmark.parse_args([*cli_args(tmp_path), "--smoke", *options])
    assert error.value.code == 2


@pytest.mark.parametrize("key,value", [
    ("completed_forwards", 7), ("block_calls", 223), ("pack_count", 6), ("restore_count", 6),
    ("velocity_updates", 3), ("spatial_tokens", 360), ("dense_gen_tokens", 3092),
    ("sparse_gen_tokens", 1844), ("retained_tokens", 183), ("sparse_enabled", False),
])
def test_smoke_rejects_incomplete_or_wrong_sparse_execution(key, value):
    summary = {**sparse_summary(), key: value}
    with pytest.raises(RuntimeError, match=key):
        benchmark.validate_smoke_result("sparse", {"action": np.zeros((32, 8))}, summary)


@pytest.mark.parametrize("action", [np.zeros((32, 7)), np.full((32, 8), np.nan), np.full((32, 8), np.inf)])
def test_smoke_rejects_invalid_action(action):
    with pytest.raises(RuntimeError, match="finite action"):
        benchmark.validate_smoke_result("sparse", {"action": action}, sparse_summary())


def test_checkpoint_snapshot_revision_without_sidecar(tmp_path):
    checkpoint = tmp_path / "snapshots" / ("a" * 40)
    checkpoint.mkdir(parents=True)
    result = benchmark.collect_provenance(str(checkpoint), tmp_path / "observation.npz")
    assert result == {"checkpoint": None, "observation": None, "checkpoint_revision": "a" * 40}


def test_smoke_validates_warmup_and_measured_inferences_and_records_provenance(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "provenance.json").write_text(json.dumps({"resolved_revision": "checkpoint-sha"}))
    np.savez(tmp_path / "observation.npz", prompt=np.array("pick up the cup"))
    (tmp_path / "observation.provenance.json").write_text(json.dumps({
        "resolved_revision": "input-sha",
        "observation_sha256": hashlib.sha256((tmp_path / "observation.npz").read_bytes()).hexdigest(),
    }))
    config = SparseWAMConfig("edge")
    runner = SimpleNamespace(config=config, mode="dense", last_summary=None)
    calls = []

    def infer(observation):
        assert observation["prompt"] == "pick up the cup"
        calls.append(runner.mode)
        runner.last_summary = sparse_summary() if runner.mode == "sparse" else config.run_metadata("dense")
        return {"action": np.full((32, 8), 0.5 if runner.mode == "sparse" else 0.0)}

    service = SimpleNamespace(sparse_wam_runner=runner, infer=infer)
    monkeypatch.setattr(benchmark, "create_service", lambda *args, **kwargs: (service, None))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(benchmark, "environment_versions", lambda: {"torch": "test"})
    checked = []
    validate = benchmark.validate_smoke_result

    def recording_validate(mode, output, summary):
        checked.append(mode)
        validate(mode, output, summary)

    monkeypatch.setattr(benchmark, "validate_smoke_result", recording_validate)
    benchmark.main([*cli_args(tmp_path), "--smoke"])
    report = json.loads((tmp_path / "report.json").read_text())
    assert calls == checked == ["dense", "sparse", "sparse", "dense"]
    assert report["smoke"]["passed"]
    assert report["smoke"]["status"] == "passed"
    assert datetime.fromisoformat(report["started_at_utc"]).utcoffset() == timedelta(0)
    assert len(report["smoke"]["checks"]) == 4
    assert report["action_shapes"] == {"dense": [32, 8], "sparse": [32, 8]}
    assert report["provenance"]["checkpoint_revision"] == "checkpoint-sha"
    assert report["provenance"]["observation"]["resolved_revision"] == "input-sha"
    assert report["action_mse_by_pair"] == [0.25]
    assert report["reproduction"]["input_files"]["observation"]["sha256"] == hashlib.sha256(
        (tmp_path / "observation.npz").read_bytes()
    ).hexdigest()
    assert report["reproduction"]["source"]["version"]
    assert report["effective_runs"]["sparse"]["checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("failure_stage", ["input", "model"])
def test_failed_smoke_replaces_old_passed_report(tmp_path, monkeypatch, failure_stage):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"smoke": {"passed": True}}))
    if failure_stage == "model":
        np.savez(tmp_path / "observation.npz", prompt=np.array("pick up the cup"))

    def fail_model(*args, **kwargs):
        running = json.loads(report_path.read_text())
        assert running["smoke"] == {"passed": False, "status": "running"}
        raise RuntimeError("model load failed")

    monkeypatch.setattr(benchmark, "create_service", fail_model)
    error_type = FileNotFoundError if failure_stage == "input" else RuntimeError
    with pytest.raises(error_type):
        benchmark.main([*cli_args(tmp_path), "--smoke"])
    report = json.loads(report_path.read_text())
    assert report["smoke"]["passed"] is False
    assert report["smoke"]["status"] == "failed"
    assert report["smoke"]["error"].startswith(f"{error_type.__name__}: ")
    assert report["reproduction"]["source"]["version"]
    assert datetime.fromisoformat(report["started_at_utc"]).utcoffset() == timedelta(0)


def test_smoke_rejects_changed_observation_before_loading_model(tmp_path, monkeypatch):
    np.savez(tmp_path / "observation.npz", prompt=np.array("changed input"))
    (tmp_path / "observation.provenance.json").write_text(json.dumps({"observation_sha256": "0" * 64}))

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("Model must not load when the input provenance is stale")

    monkeypatch.setattr(benchmark, "create_service", unexpected_model_load)
    with pytest.raises(ValueError, match="SHA256 does not match"):
        benchmark.main([*cli_args(tmp_path), "--smoke"])
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["smoke"]["passed"] is False
    assert report["smoke"]["status"] == "failed"
