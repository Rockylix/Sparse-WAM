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


def joint_graph_summary():
    return {
        "backbone": "fastwam-joint", "mode": "sparse", "sparse_enabled": True, "execution": "compile-graph",
        "dense_forwards": 1, "sparse_forwards": 9, "dense_steps_executed": [0],
        "completed_steps": list(range(10)), "profile_blocks": list(range(30)), "sparse_steps": list(range(1, 10)),
        "dense_total_tokens": 326, "fixed_tokens_per_future": 56, "core_tokens_per_future": 19,
        "stable_tokens_shared": 37, "condition_kv_cache_tokens": 98, "sparse_query_tokens": 144,
        "sparse_key_value_tokens": 242, "single_pack_per_request": True, "background_cache_source_step": 0,
        "background_cache_update_steps": list(range(1, 10)), "cross_attention_kv_reused_from_step0": True,
        "fused_step0_profile_used": True, "fused_step0_profile_error": None, "torch_compile_requested": True,
        "torch_compile_boundary": "step1_N_complete_B0_B29_sparse_stack", "torch_compile_fallback_error": None,
        "cuda_graph_requested": True, "cuda_graph_fallback_error": None, "cuda_graph_capture_count": 1,
        "cuda_graph_capture_validation": {"relative_l2": 0.0, "max_absolute_error": 0.0},
    }


@pytest.mark.parametrize("override,delta,match", [
    ({"completed_steps": [0]}, 9, "completed_steps"),
    ({"sparse_key_value_tokens": 144}, 9, "sparse_key_value_tokens"),
    ({"fused_step0_profile_error": "fallback"}, 9, "fused_step0_profile_error"),
    ({"cuda_graph_capture_validation": {"relative_l2": float("nan")}}, 9, "relative_l2"),
    ({"cuda_graph_capture_count": 2}, 9, "capture"),
    ({}, 18, "replay delta"),
])
def test_joint_requires_verified_k56_graph_execution(override, delta, match):
    summary = joint_graph_summary()
    action = np.zeros((32, 7))
    with pytest.raises(RuntimeError, match=match):
        benchmark.validate_joint_result("sparse", action, {**summary, **override}, "compile-graph", delta, warmed=True)


def test_joint_reference_is_compared_per_mode_in_libero_units(tmp_path):
    path = tmp_path / "reference.npz"
    dense, sparse = np.ones((32, 7)), np.zeros((32, 7))
    np.savez(path, dense_action=dense, sparse_action=sparse)
    references = benchmark.read_reference_actions(path)
    assert benchmark.compare_reference(dense, references["dense"], "dense", 0)["max_absolute_error"] == 0
    with pytest.raises(RuntimeError, match="reference max absolute error"):
        benchmark.compare_reference(dense, references["sparse"], "sparse", 0.25)
    with pytest.raises(SystemExit):
        benchmark.parse_args([*cli_args(tmp_path), "--reference-actions", str(path)])


def test_joint_measured_replays_are_deltas_across_dense_and_sparse_pairs(tmp_path, monkeypatch):
    np.savez(tmp_path / "observation.npz", task_description="move the bowl")
    dense = np.zeros((32, 7), dtype=np.float32)
    sparse = np.full((32, 7), 0.25, dtype=np.float32)
    reference = tmp_path / "reference.npz"
    proprio = np.zeros((1, 8), dtype=np.float32)
    blocks = np.arange(6)
    np.savez(reference, dense_normalized=dense, sparse_normalized=sparse, proprio=proprio, selected_blocks=blocks)
    runner = SimpleNamespace()
    replay_total = 0
    preparations = []

    def prepare_observation(observation):
        preparations.append(observation["task_description"])
        return {"seed": 42, "proprio": benchmark.torch.from_numpy(proprio)}

    def infer_prepared(sample):
        nonlocal replay_total
        assert sample["seed"] == 42
        if runner.mode == "sparse":
            replay_total += 9
            runner.last_summary = {**joint_graph_summary(), "cuda_graph_replay_count": replay_total,
                                   "selected_blocks_ranked": blocks.tolist()}
            return {"action": benchmark.torch.from_numpy(sparse)}
        runner.last_summary = runner.config.run_metadata("dense")
        return {"action": benchmark.torch.from_numpy(dense)}

    service = SimpleNamespace(sparse_wam_runner=runner,
                              prepare_observation=prepare_observation, infer_prepared=infer_prepared)
    monkeypatch.setattr(benchmark, "create_service", lambda *args, **kwargs: (service, None))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(benchmark, "environment_versions", lambda: {})
    benchmark.main([*cli_args(tmp_path), "--backbone", "fastwam-joint", "--execution", "compile-graph",
                    "--reference-actions", str(reference), "--reference-atol", "0", "--timing-scope", "model",
                    "--warmup", "2", "--repeats", "2"])
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["cuda_graph_replays_by_round"] == {"dense": [0, 0], "sparse": [9, 9]}
    assert report["effective_runs"]["sparse"]["cuda_graph_replay_count"] == 36
    assert report["reference_actions"]["units"] == "normalized model output"
    assert report["reference_actions"]["errors_by_round"]["sparse"][-1]["max_absolute_error"] == 0
    assert report["reference_actions"]["fixture_checks"] == ["proprio", "selected_blocks"]
    assert preparations == ["move the bowl"]
