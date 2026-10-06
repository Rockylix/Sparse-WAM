from types import SimpleNamespace as NS

import pytest
import torch

from sparse_wam.nano_packing import reuse_nano_packing


class FakeModel:
    def __init__(self):
        self.config = NS(video_temporal_causal=False, sound_gen=False)
        self.builds = 0

    def _pack_input_sequence(self, plans, texts, clean, timestep, **flags):
        self.builds += 1
        return NS(
            vision=NS(timesteps=timestep.flatten()[:1].repeat(4), tokens=clean.x0_tokens_vision),
            action=NS(timesteps=timestep.flatten()[:1].repeat(2), tokens=clean.x0_tokens_action),
            rope=torch.arange(6),
            condition_mask=torch.tensor([1, 0, 0]),
            texts=texts,
            flags=flags,
        )

    def _get_velocity(self, *, noise_x, timestep, text_tokens, sequence_plans, gen_data_clean, **flags):
        packed = self._pack_input_sequence(sequence_plans, text_tokens, gen_data_clean, timestep, **flags)
        # Mirror the critical native contract: update noisy inputs AFTER packing.
        packed.vision.tokens = [noise_x[0]]
        packed.action.tokens = [noise_x[0] + 1]
        return packed


def sample():
    return dict(
        noise_x=[torch.ones(4)],
        timestep=torch.tensor([999.0]),
        text_tokens=[[1, 2]],
        sequence_plans=[NS(has_sound=False)],
        gen_data_clean=NS(
            batch_size=1,
            num_vision_items_per_sample=None,
            x0_tokens_vision=[torch.zeros(4)],
            x0_tokens_action=[torch.zeros(2)],
        ),
    )


def test_only_velocity_packing_is_cached_and_inputs_still_change():
    model, data = FakeModel(), sample()
    with reuse_nano_packing(model) as stats:
        prep = model._pack_input_sequence(
            data["sequence_plans"], data["text_tokens"], data["gen_data_clean"], data["timestep"]
        )
        first = model._get_velocity(**data)
        assert first is not prep
        data.update(timestep=torch.tensor([624.0]), noise_x=[torch.full((4,), 7.0)])
        second = model._get_velocity(**data)
        assert first is second
        assert torch.equal(second.vision.tokens[0], data["noise_x"][0])
        assert (second.vision.timesteps == 624).all()
        assert (second.action.timesteps == 624).all()
        assert torch.equal(second.rope, torch.arange(6))
        assert torch.equal(second.condition_mask, torch.tensor([1, 0, 0]))
    assert stats == dict(enabled=True, calls=3, builds=2, hits=1, bypasses=1)
    assert "_get_velocity" not in vars(model) and "_pack_input_sequence" not in vars(model)
    with reuse_nano_packing(model) as next_stats:
        assert model._get_velocity(**data) is not first
    assert next_stats["hits"] == 0


@pytest.mark.parametrize("change", ["text", "skip", "offset", "end", "shape", "clean", "plans"])
def test_distinct_metadata_never_shares_a_template(change):
    model, data = FakeModel(), sample()
    with reuse_nano_packing(model) as stats:
        first = model._get_velocity(**data)
        if change == "text":
            data["text_tokens"] = [[4, 5]]
        elif change == "skip":
            data["skip_text_tokens"] = True
        elif change == "offset":
            data["initial_mrope_temporal_offset"] = 4
        elif change == "end":
            data["include_end_of_generation_token"] = True
        elif change == "shape":
            data["gen_data_clean"].x0_tokens_vision = [torch.zeros(8)]
        elif change == "clean":
            data["gen_data_clean"] = sample()["gen_data_clean"]
        elif change == "plans":
            data["sequence_plans"] = [NS(has_sound=False)]
        assert model._get_velocity(**data) is not first
    assert stats["builds"] == 2 and stats["hits"] == 0


@pytest.mark.parametrize("case", ["batch", "per_frame", "multi_item", "sound", "causal"])
def test_unsupported_shapes_fall_back(case):
    model, data = FakeModel(), sample()
    if case == "batch":
        data["gen_data_clean"].batch_size = 2
    elif case == "per_frame":
        data["timestep"] = torch.tensor([[999.0, 749.0]])
    elif case == "multi_item":
        data["gen_data_clean"].num_vision_items_per_sample = [2]
    elif case == "sound":
        model.config.sound_gen = True
    else:
        model.config.video_temporal_causal = True
    with reuse_nano_packing(model) as stats:
        first = model._get_velocity(**data)
        assert model._get_velocity(**data) is not first
    assert stats["hits"] == 0 and stats["bypasses"] == 2


def test_exception_restores_existing_instance_overrides():
    model = FakeModel()
    old = model._get_velocity
    model._get_velocity = old
    with pytest.raises(RuntimeError):
        with reuse_nano_packing(model):
            model._get_velocity(**sample())
            raise RuntimeError("failure")
    assert model._get_velocity is old
    assert "_pack_input_sequence" not in vars(model)


def test_disabled_is_a_noop():
    model = FakeModel()
    with reuse_nano_packing(model, enabled=False) as stats:
        assert "_get_velocity" not in vars(model)
        model._get_velocity(**sample())
    assert stats["enabled"] is False and stats["hits"] == 0
