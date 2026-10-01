"""Exercise the actual controller hooks and sampler across eight native-ABI forwards."""

from types import SimpleNamespace

import pytest
import torch

from sparse_wam.config import PaperConfig
from sparse_wam.controller import SparseWAMController
from sparse_wam.velocity import VelocityCacheSampler


class TestPacking:
    __test__ = False

    def gen(self, pack):
        return pack["full_only_seq"]

    def replace_gen(self, pack, gen):
        return {**pack, "full_only_seq": gen}

    def slice(self, pack, rope, selected):
        def cut(p):
            return {**p, "full_only_seq": self.gen(p)[selected], "_num_full_tokens": selected.numel()}

        return cut(pack), tuple(cut(component) for component in rope)


class RecordingBackend:
    hook_name = "_robolab_version1_controller"

    def __init__(self, config, fail_at=None):
        self.config = config
        self.calls = []
        self.fail_at = fail_at
        self.exited = False

    def enter(self, layers):
        pass

    def exit(self):
        self.exited = True

    def run(self, layer, hidden, mask, rope, memory, gen_only, geometry, profile):
        if len(self.calls) == self.fail_at:
            raise RuntimeError("injected decoder failure")
        x = hidden["full_only_seq"]
        # Original positional identifiers must survive every sparse gather.
        assert torch.equal(x[:, 1], rope[0]["full_only_seq"][:, 0])
        self.calls.append((profile, len(x), hidden["_num_causal_tokens"]))
        output = x.clone()
        output[:, 0] += 1
        scores = torch.arange(self.config.spatial_tokens).float().repeat(8, 1) / 1000 if profile else None
        return {**hidden, "full_only_seq": output}, {}, None, scores


class FakeNativeNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_patch_size = (1, 1, 1)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=SimpleNamespace(hidden_size=config.hidden_size))
                    for _ in range(config.layers)
                ]
            )
        )

    def layout(self, und=3):
        spatial = self.config.spatial_tokens
        return SimpleNamespace(
            attn_modes=["causal", "full"],
            split_lens=[und, 9 * spatial + 33],
            vision=SimpleNamespace(
                sequence_indexes=torch.arange(und, und + 9 * spatial), token_shapes=[(9, 1, spatial)]
            ),
            action=SimpleNamespace(
                sequence_indexes=torch.arange(und + 9 * spatial, und + 9 * spatial + 33),
                token_shapes=[(33,)],
                condition_mask=[torch.tensor([True] + [False] * 32)],
            ),
        )

    def forward(self, packed_seq, memory=None, und_only=False, value=0.0):
        if und_only:
            return {}
        c = self.language_model.model._robolab_version1_controller
        n = packed_seq.split_lens[1]
        gen = torch.stack((torch.full((n,), value), torch.arange(n).float()), dim=1)
        hidden = {
            "full_only_seq": gen,
            "causal_seq": torch.zeros(packed_seq.split_lens[0], 2),
            "_num_full_tokens": n,
            "_num_causal_tokens": packed_seq.split_lens[0],
        }
        rope = ({**hidden, "full_only_seq": gen[:, 1:].clone()},) * 2
        c.begin_stack(hidden_states=hidden, position_embeddings=rope, natten_metadata_list=None)
        for i, layer in enumerate(self.language_model.model.layers):
            hidden, _, _ = c.run_layer(
                block=i,
                decoder_layer=layer,
                hidden_states=hidden,
                attention_mask=None,
                memory_value=None,
                gen_only=True,
            )
        hidden = c.end_stack(hidden)
        x = hidden["full_only_seq"][:, 0]
        spatial = self.config.spatial_tokens
        return {
            "preds_vision": [x[: 9 * spatial].reshape(1, 9, 1, spatial)],
            "preds_action": [x[9 * spatial :]],
        }


@pytest.mark.parametrize("tier", ["edge", "nano"])
def test_only_conditional_step0_dense_and_velocity_cache_is_request_local(tier):
    config = PaperConfig(tier)
    net = FakeNativeNet(config)
    backend = RecordingBackend(config)
    c = SparseWAMController(net, config, backend=backend, packing=TestPacking())
    outputs = []

    def solver(velocity, noise, **kwargs):
        for step in range(4):
            outputs.append(velocity(noise, torch.tensor([[step]]))[0])
        return outputs[-1]

    def cfg_velocity(noise, timestep):
        step = int(timestep.item())
        cond = net(net.layout(7), value=step * 10.0)
        uncond = net(net.layout(2), value=step * 10.0 + 2.0)

        # Native order is vision then action, and native CFG is left intact.
        def flatten(out):
            return torch.cat((out["preds_vision"][0].flatten(), out["preds_action"][0]))

        v_c, v_u = flatten(cond), flatten(uncond)
        return [v_u + 3 * (v_c - v_u)]

    with c:
        net(net.layout(), und_only=True)  # prefill must not consume a CFG slot
        sampler = VelocityCacheSampler(solver, c)
        sampler(cfg_velocity, [torch.empty(0)], num_steps=4)
    summary = c.finish()
    assert summary["pack_count"] == summary["restore_count"] == 7
    assert sum(profile for profile, _, _ in backend.calls) == config.layers
    dense_len = 9 * config.spatial_tokens + 33
    sparse_len = config.spatial_tokens + 8 * 184 + 33
    assert [length for _, length, _ in backend.calls[: config.layers]] == [dense_len] * config.layers
    assert all(length == sparse_len for _, length, _ in backend.calls[config.layers :])
    assert {und for _, _, und in backend.calls} == {2, 7}
    mask = c.plan.keep
    spatial = config.spatial_tokens
    for step, flat in enumerate(outputs):
        future = flat[: 9 * spatial].reshape(9, spatial)[1:]
        assert (future[~mask] == config.layers).all()  # full conditional background, frozen
        assert (future[mask] == config.layers + step * 10 - 4).all()
        assert (flat[-33:] == config.layers + step * 10 - 4).all()  # actions always fresh
    assert not hasattr(net.language_model.model, backend.hook_name)
    assert backend.exited
    with pytest.raises(RuntimeError, match="reused"):
        with c:
            pass


def test_failure_cleans_hooks_and_allows_a_fresh_request():
    config = PaperConfig("edge")
    net = FakeNativeNet(config)
    backend = RecordingBackend(config, fail_at=3)
    c = SparseWAMController(net, config, backend=backend, packing=TestPacking())
    with pytest.raises(RuntimeError, match="injected"):
        with c:
            net(net.layout())
    assert not net._forward_hooks and not net._forward_pre_hooks
    assert not hasattr(net.language_model.model, backend.hook_name)
    assert backend.exited
    fresh = SparseWAMController(net, config, backend=RecordingBackend(config), packing=TestPacking())
    with fresh:
        net(net.layout())
    assert fresh.forward_count == 1


def test_layout_changes_fail_before_sparse_layer_execution():
    config = PaperConfig("edge")
    net = FakeNativeNet(config)
    c = SparseWAMController(net, config, backend=RecordingBackend(config), packing=TestPacking())
    with c:
        net(net.layout())
        changed = net.layout()
        changed.action.condition_mask[0][0] = False
        with pytest.raises(ValueError, match="layout changed"):
            net(changed)


@pytest.mark.parametrize("tier", ["edge", "nano"])
def test_paper_runner_refreshes_background_across_successive_chunks(monkeypatch, tier):
    import sparse_wam.runtime as runtime

    config = PaperConfig(tier)
    net = FakeNativeNet(config)
    controllers = []

    def build_controller(model_net, settings):
        controller = SparseWAMController(
            model_net, settings, backend=RecordingBackend(settings), packing=TestPacking()
        )
        controllers.append(controller)
        return controller

    monkeypatch.setattr(runtime, "SparseWAMController", build_controller)

    class UniPCSampler:
        """Deterministic solver double; real hook/CFG/cache execution occurs below."""

        def __call__(self, velocity, noise, **kwargs):
            for step in range(kwargs["num_steps"]):
                result = velocity(noise, torch.tensor([[step]]))
            return result

    native_sampler = UniPCSampler()

    def generate(batch, *, sampler, num_steps, guidance, shift):
        assert (num_steps, guidance, shift) == (4, 3.0, 5.0)
        assert isinstance(batch["caption"][0], str)
        net(net.layout(), und_only=True)

        def velocity(noise, timestep):
            value = batch["offset"] + timestep.item() * 10
            cond = net(net.layout(7), value=value)
            uncond = net(net.layout(2), value=value + 2)

            def flatten(out):
                return torch.cat((out["preds_vision"][0].flatten(), out["preds_action"][0]))

            c, u = flatten(cond), flatten(uncond)
            return [u + guidance * (c - u)]

        result = sampler(velocity, [torch.empty(0)], num_steps=num_steps, shift=shift)[0]
        count = 9 * config.spatial_tokens
        return {
            "vision": [result[:count].reshape(1, 9, 1, config.spatial_tokens)],
            "action": [result[count:]],
        }

    model = SimpleNamespace(
        net=net,
        sampler=native_sampler,
        input_caption_key="caption",
        config=SimpleNamespace(rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc")),
        generate_samples_from_batch=generate,
    )
    runner = runtime.PaperRunner(model, config)
    for offset in (0.0, 100.0):
        output = runner({"caption": [{"instruction": "pick up the cup"}], "offset": offset})
        background = output["vision"][0][0, 1:, 0][~controllers[-1].plan.keep]
        assert (background == offset + config.layers).all()
        assert runner.last_summary["velocity_updates"] == 4
        assert runner.last_summary["pack_count"] == 7
        assert model.sampler is native_sampler
        assert not net._forward_hooks and not net._forward_pre_hooks
    assert controllers[0] is not controllers[1]
