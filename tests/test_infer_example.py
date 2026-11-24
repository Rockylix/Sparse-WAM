"""Exercise the public offline example's data boundary without loading a model."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "infer_edge_example", Path(__file__).resolve().parents[1] / "examples" / "infer_edge.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def arguments(tmp_path):
    return ["--checkpoint-path", "local-checkpoint", "--observation", str(tmp_path / "input.npz"),
            "--output", str(tmp_path / "action.npz")]


def test_example_preserves_observation_and_saves_actions_with_provenance(tmp_path, monkeypatch):
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    np.savez(tmp_path / "input.npz", prompt=np.array("pick up the cup"), image=pixels)
    expected = np.zeros((32, 8), dtype=np.float32)

    def create_service(config, **kwargs):
        assert config.backbone == "edge" and config.execution == "eager"
        assert kwargs["checkpoint"] == "local-checkpoint"

        def infer(obs):
            assert obs["prompt"] == "pick up the cup"
            np.testing.assert_array_equal(obs["image"], pixels)
            return {"action": expected}

        return SimpleNamespace(infer=infer, sparse_wam_runner=SimpleNamespace(last_summary={"mode": "sparse"})), \
            SimpleNamespace(checkpoint_path="local-checkpoint")

    monkeypatch.setattr(example, "create_service", create_service)
    example.main(arguments(tmp_path))
    with np.load(tmp_path / "action.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["action"], expected)
    report = json.loads((tmp_path / "action.json").read_text())
    assert report["summary"]["checkpoint"] == report["checkpoint"] == "local-checkpoint"
    assert report["action_shape"] == [32, 8]
    assert report["reproduction"]["input_files"]["observation"]["sha256"]
    with pytest.raises(SystemExit):
        example.main(arguments(tmp_path))


def test_example_rejects_pickled_observation_before_loading_model(tmp_path, monkeypatch):
    np.savez(tmp_path / "input.npz", prompt=np.array({"unsafe": "object"}, dtype=object))

    def unexpected_model(*args, **kwargs):
        pytest.fail("Malformed input must not load a model")

    monkeypatch.setattr(example, "create_service", unexpected_model)
    with pytest.raises(ValueError, match="Object arrays"):
        example.main(arguments(tmp_path))
    assert not (tmp_path / "action.npz").exists()
