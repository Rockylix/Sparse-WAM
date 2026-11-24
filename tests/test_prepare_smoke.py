"""The recorded sample must retain the official image layout and safe NPZ schema."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("prepare_smoke", Path(__file__).parents[1] / "tools/prepare_smoke.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.mark.parametrize("filename", ["observation.npz", "observation", "observation.data"])
def test_observation_layout_and_npz_schema(tmp_path, filename):
    image = pytest.importorskip("PIL.Image")
    frames = [image.new("RGB", (640, 360), color) for color in ("red", "green", "blue")]
    path = tmp_path / filename
    prepare.save_observation(frames, path)
    with np.load(path, allow_pickle=False) as observation:
        pixels = observation["observation/image"]
        assert pixels.shape == (540, 640, 3)
        assert pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(pixels[400, 0], [0, 128, 0])
        np.testing.assert_array_equal(pixels[400, 400], [0, 0, 255])
        assert observation["observation/joint_position"].shape == (7,)
        assert observation["observation/gripper_position"].item() == 0.0
        assert observation["prompt"].item() == prepare.PROMPT


def test_observation_rejects_wrong_camera_layout():
    image = pytest.importorskip("PIL.Image")
    frames = [image.new("RGB", size) for size in [(640, 360), (640, 360), (320, 180)]]
    with pytest.raises(ValueError, match="three 640x360"):
        prepare.compose_observation(frames)


def test_metadata_comes_from_literal_official_example():
    source = '''
parser.add_argument("--prompt", default="Put the cup down.")
request = {"action_params": {"observation": {
    "observation/joint_position": [1, 2, 3, 4, 5, 6, 7],
    "observation/gripper_position": 0.5,
}}}
raise RuntimeError("The example must not execute")
'''
    prompt, joints, gripper = prepare.read_example_metadata(source)
    assert prompt == "Put the cup down."
    np.testing.assert_array_equal(joints, [1, 2, 3, 4, 5, 6, 7])
    assert gripper == 0.5


def test_metadata_rejects_computed_state():
    source = '''
parser.add_argument("--prompt", default="Put the cup down.")
request = {"observation": dict()}
'''
    with pytest.raises(ValueError):
        prepare.read_example_metadata(source)


def test_checkpoint_prepares_native_vae_in_hub_cache(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    cache_file = tmp_path / "hub" / prepare.VAE_FILENAME
    cache_file.parent.mkdir()
    cache_file.touch()
    calls = []

    def snapshot_download(**kwargs):
        kwargs["local_dir"].mkdir(parents=True)

    def hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(cache_file)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha=prepare.MODEL_REVISION)),
        snapshot_download=snapshot_download,
        hf_hub_download=hf_hub_download,
    ))
    checkpoint = tmp_path / "checkpoint"
    prepare.prepare_checkpoint(checkpoint, prepare.MODEL_REVISION)
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    assert calls == [{
        "repo_id": prepare.VAE_REPOSITORY,
        "revision": prepare.VAE_REVISION,
        "filename": prepare.VAE_FILENAME,
    }]
    assert provenance["native_vae"]["path"] == str(cache_file.resolve())
    assert provenance["native_vae"]["resolved_revision"] == prepare.VAE_REVISION
