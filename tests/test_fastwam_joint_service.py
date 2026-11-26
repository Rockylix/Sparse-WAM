"""The LIBERO policy contract can be validated on CPU without assets or a simulator."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparse_wam import SparseWAMConfig
from sparse_wam.backends.fastwam_joint import JointRunner, JointService, resolve_settings
from sparse_wam.backends.libero import LiberoProcessor, PROMPT


def test_joint_settings_resolve_native_paths_and_explicit_checkpoint(tmp_path):
    config = SparseWAMConfig("fastwam-joint")
    (tmp_path / "cache").mkdir()
    for name in ("explicit.pt", "vae.pth", "stats.json"):
        (tmp_path / name).touch()
    overrides = dict(checkpoint_path="not_used.pt", vae_path="vae.pth",
                     dataset_stats_path="stats.json", text_cache_dir="cache")
    settings = resolve_settings(config, tmp_path / "explicit.pt", overrides, tmp_path / "assets.json", 42)
    assert settings.checkpoint_path == tmp_path / "explicit.pt"
    assert settings.text_cache_dir == tmp_path / "cache"
    with pytest.raises(ValueError, match="sampler"):
        resolve_settings(config, None, {"sampler": "unipc"}, None, 42)


def test_libero_chunk_uses_camera_order_rotation_stats_and_text_padding(tmp_path):
    stats = {}
    for key, size in (("action", 7), ("state", 8)):
        stats[key] = {"default": {"global_min": [0] * size, "global_max": [2] * size}}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    processor = LiberoProcessor(stats_path, tmp_path, "cpu", torch.bfloat16)
    instruction = "put the cup down"
    prompt = PROMPT.format(task=instruction)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    text = torch.ones((128, 4096), dtype=torch.bfloat16)
    mask = torch.arange(128) < 4
    torch.save(dict(prompt=prompt, context=text, mask=mask),
               tmp_path / f"{digest}.t5_len128.wan22ti2v5b.pt")
    camera = np.zeros((224, 224, 3), dtype=np.uint8)
    camera[0, 0] = 255
    observation = {
        "agentview_image": camera, "robot0_eye_in_hand_image": np.full_like(camera, 255),
        "robot0_eef_pos": np.array([0.0, 1.0, 2.0]), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        "robot0_gripper_qpos": np.array([1., 1.]), "prompt": instruction,
    }
    runner = JointRunner(SimpleNamespace(infer_action=lambda **kw: {"action": torch.zeros(32, 7)}),
                         SparseWAMConfig("fastwam-joint", "compile-graph"), "dense", 42, None)
    service = JointService(runner, processor)
    prepared = service.prepare_observation(observation)
    assert prepared["seed"] == 42 and prepared["num_inference_steps"] == 10
    assert prepared["input_image"].shape == (1, 3, 224, 448)
    assert (prepared["input_image"][:, :, 223, 223] == 1).all()
    assert (prepared["input_image"][:, :, 0, 0] == -1).all()
    assert (prepared["input_image"][:, :, :, 224:] == 1).all()
    torch.testing.assert_close(prepared["proprio"], torch.tensor([[-1., 0., 1., -1., -1., -1., 0., 0.]]))
    assert prepared["prompt"] is None and prepared["context_mask"].all()
    assert (prepared["context"][:, :4] == 1).all() and (prepared["context"][:, 4:] == 0).all()
    action = service.infer(observation)["action"]
    assert action.shape == (32, 7)
    assert np.all(action[:, :6] == 1) and np.all(action[:, 6] == -1)
    assert runner.last_summary["execution"] == "eager"
    assert runner.last_summary["dense_forwards"] == 10


@pytest.mark.parametrize("execution,compile_requested,graph", [
    ("eager", False, False), ("compile", True, False), ("compile-graph", True, True),
])
def test_joint_switches_inner_execution_only(execution, compile_requested, graph):
    class Controller:
        def __init__(self, **options):
            self.options = options

        def __enter__(self):
            assert self.options["torch_compile"] == compile_requested
            assert self.options["cuda_graph"] == graph
            assert self.options["num_steps"] == 10

        def __exit__(self, *args):
            pass

        def summary(self):
            return {"sparse_query_tokens": 144, "sparse_key_value_tokens": 242}

    model = SimpleNamespace(mot=object(), infer_action=lambda: {"action": torch.ones(32, 7)})
    runner = JointRunner(model, SparseWAMConfig("fastwam-joint", execution), "sparse", 42, Controller)
    runner({})
    assert runner.last_summary["sparse_query_tokens"] == 144
    assert runner.last_summary["sparse_forwards"] == 9
    runner.mode = "dense"
    runner({})
    assert runner.last_summary["sparse_forwards"] == 0
    assert runner.last_summary["execution"] == "eager"


def test_checkpoint_requires_all_inference_weights_and_accepts_spectral_plan_extras():
    from sparse_wam.backends.fastwam_joint import load_joint_weights

    model = SimpleNamespace(mot=torch.nn.Linear(2, 2), proprio_encoder=torch.nn.Linear(1, 2))
    mot = model.mot.state_dict()
    mot["plan_adapters.9.gate"] = torch.empty(1)
    weights = {"mot": mot, "proprio_encoder": model.proprio_encoder.state_dict()}
    load_joint_weights(model, weights)
    mot.pop("bias")
    with pytest.raises(RuntimeError, match="bias"):
        load_joint_weights(model, weights)
