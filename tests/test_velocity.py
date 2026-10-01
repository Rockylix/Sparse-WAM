from types import SimpleNamespace

import pytest
import torch

from sparse_wam.velocity import VelocityCacheSampler, expand_future_mask, merge_future


def test_patch_expansion_trims_padding_without_resizing():
    tokens = torch.zeros(8, 4, dtype=torch.bool)
    tokens[:, 3] = True
    mask = expand_future_mask(tokens, (2, 2), (3, 3), (1, 2, 2))
    assert torch.equal(mask, expand_future_mask(tokens, (2, 2), (3, 3), 2))
    assert mask.sum() == 8
    assert mask[:, 2, 2].all()
    with pytest.raises(ValueError, match="geometry"):
        expand_future_mask(tokens, (2, 2), (5, 5), (1, 2, 2))


@pytest.mark.parametrize("leading", [(2,), (1, 2)])
def test_velocity_equation_preserves_current_l0_and_selected_future(leading):
    current = torch.full((*leading, 9, 3, 3), 7.0)
    cached = torch.full_like(current, 2.0)
    mask = torch.zeros(8, 3, 3, dtype=torch.bool)
    mask[:, 1, 2] = True
    result = merge_future(current, cached, mask)
    assert (result[..., 0, :, :] == 7).all()
    assert (result[..., 1:, 1, 2] == 7).all()
    assert (result[..., 1:, 0, 0] == 2).all()
    assert (current == 7).all()  # no alias mutation of model predictions


def test_sampler_uses_guided_cache_before_solver_and_preserves_actions():
    mask = torch.zeros(8, 2, 2, dtype=torch.bool)
    mask[:, 0, 0] = True
    controller = SimpleNamespace(vision_shape=(1, 9, 2, 2), velocity_mask=mask)
    observed = []

    def native_solver(velocity, noise, **kwargs):
        # A deterministic solver probe observes the values actually supplied to it.
        state = noise
        for step in range(4):
            update = velocity(state, torch.tensor([[step]]))
            observed.append(update[0].clone())
            state = [state[0] + update[0]]
        return state

    def guided_velocity(noise, timestep):
        step = int(timestep.item())
        return [torch.cat((torch.full((36,), float(step + 1)), torch.tensor([100.0 + step])))]

    sampler = VelocityCacheSampler(native_solver, controller)
    result = sampler(guided_velocity, [torch.zeros(37)], num_steps=4, shift=5.0)
    for step, flat in enumerate(observed):
        vision = flat[:36].reshape(1, 9, 2, 2)
        assert (vision[:, 0] == step + 1).all()
        assert (vision[:, 1:, 0, 0] == step + 1).all()
        assert (vision[:, 1:, 1, 1] == 1).all()
        assert flat[-1] == 100 + step
    assert result[0][-1] == 406
    with pytest.raises(RuntimeError, match="chunks"):
        sampler(guided_velocity, [torch.zeros(37)], num_steps=4)
