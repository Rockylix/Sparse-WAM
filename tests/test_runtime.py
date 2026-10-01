import json
from types import SimpleNamespace

import pytest
import torch

from sparse_wam.config import PaperConfig
from sparse_wam.runtime import PaperRunner, normalize_caption


def test_structured_caption_preserves_task_text_without_mutating_input():
    original = {"ai_caption": [{"instruction": "把香蕉放进碗里", "views": "three"}], "image": object()}
    result = normalize_caption(original, "ai_caption")
    assert json.loads(result["ai_caption"][0]) == original["ai_caption"][0]
    assert isinstance(original["ai_caption"][0], dict)
    assert result["image"] is original["image"]
    assert normalize_caption(result, "ai_caption")["ai_caption"] == result["ai_caption"]


@pytest.mark.parametrize(
    "override",
    [
        {"guidance": 1.0},
        {"num_steps": 8},
        {"shift": 1.0},
        {"guidance_interval": [0, 1]},
        {"normalize_cfg": True},
    ],
)
def test_reject_incompatible_generation_schedule(override):
    with pytest.raises(ValueError):
        PaperConfig("edge").validate_generation(override)


def test_dense_control_metadata_and_failed_result_do_not_claim_sparse_execution():
    output = {"action": [torch.ones(32, 7)], "vision": [torch.ones(1, 9, 2, 2)]}
    calls = []

    def generate(batch, **kwargs):
        calls.append((batch, kwargs))
        return output

    model = SimpleNamespace(
        sampler=type("UniPCSampler", (), {})(),
        input_caption_key="caption",
        config=SimpleNamespace(rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc")),
        generate_samples_from_batch=generate,
    )
    runner = PaperRunner(model, PaperConfig("nano", "compile"), mode="dense")
    batch = {"caption": [{"instruction": "pick up the cup"}]}
    assert runner(batch, seed=[1]) is output
    assert calls[0][1] == {"guidance": 3.0, "num_steps": 4, "shift": 5.0, "seed": [1]}
    assert json.loads(calls[0][0]["caption"][0]) == batch["caption"][0]
    assert runner.last_summary["dense_forwards"] == 8
    assert runner.last_summary["sparse_forwards"] == 0
    assert runner.last_summary["scorer"] is None
    assert runner.last_summary["execution"] == "eager"
    output["action"] = []
    with pytest.raises(RuntimeError, match="action"):
        runner(batch)
    assert runner.last_summary is None
