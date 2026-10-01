import ast
from pathlib import Path

import pytest
import torch

from sparse_wam.scoring import profile_from_main_lse
from sparse_wam.selection import select_tokens


def original_edge_selector():
    """Execute only the original pure selector, avoiding CUDA/framework imports."""
    path = Path(__file__).parents[1] / "vendor/edge/cosmos_framework/inference/edge_core_stable.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_select_mask", "build_core_stable_plan")
    ]
    source = "from __future__ import annotations\nimport math\n" + "\n".join(
        ast.unparse(n) for n in functions
    )
    namespace = dict(
        PROFILE_BLOCKS=tuple(range(28)),
        CORE_BLOCK_COUNT=6,
        CORE_TOKEN_BUDGET=80,
        STABLE_TOKEN_BUDGET=104,
        STABLE_CV_PENALTY=1.0,
    )
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["build_core_stable_plan"]


def test_selector_matches_pinned_edge_on_non_degenerate_input():
    profiles = torch.rand(28, 8, 340, generator=torch.Generator().manual_seed(314))
    expected = original_edge_selector()(
        torch=torch, profile_records=[{"block": b, "profiles": profiles[b]} for b in range(28)]
    )
    actual = select_tokens(profiles)
    assert torch.equal(actual.core, expected["core_masks"])
    assert torch.equal(actual.stable, expected["stable_mask"])
    assert torch.equal(actual.keep, expected["execution_mask"])


@pytest.mark.parametrize("layers,spatial", [(28, 340), (36, 360)])
def test_frame_local_motion_and_shared_anchors(layers, spatial):
    # Each frame has its own hotspot, with a shared low-variance background.
    profiles = torch.full((layers, 8, spatial), 0.001)
    for frame in range(8):
        profiles[:, frame, frame * 20 : frame * 20 + 80] = 0.3
    profiles[:, :, -104:] = 0.02
    plan = select_tokens(profiles)
    assert torch.all(plan.core.sum(-1) == 80)
    assert plan.stable.sum() == 104
    assert torch.all(plan.keep.sum(-1) == 184)
    assert not (plan.core & plan.stable).any()
    assert not torch.equal(plan.core[0], plan.core[-1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_profile_rejected(bad):
    raw = torch.ones(28, 8, 340)
    raw[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        select_tokens(raw)


def test_zero_mass_has_finite_exact_budget():
    plan = select_tokens(torch.zeros(36, 8, 360))
    assert torch.isfinite(plan.quality).all()
    assert torch.all(plan.keep.sum(-1) == 184)


@pytest.mark.parametrize("heads,kv_heads,und", [(4, 4, 7), (8, 2, 3), (4, 1, 19)])
def test_pilot_matches_full_visible_key_softmax(heads, kv_heads, und):
    spatial, width = 5, 6
    gen = 9 * spatial + 33
    rng = torch.Generator().manual_seed(71)
    q = torch.randn(gen, heads, width, generator=rng)
    k = torch.randn(gen, kv_heads, width, generator=rng)
    # Strong UND keys make future-only softmax a detectably incorrect reference.
    ku = torch.randn(und, kv_heads, width, generator=rng) * 2
    all_k = torch.cat((ku, k)).repeat_interleave(heads // kv_heads, dim=1)
    action_start = 9 * spatial + 1
    logits = torch.einsum("qhd,khd->qhk", q[action_start:] * 0.25, all_k)
    lse = logits.logsumexp(-1)[None]
    profile = profile_from_main_lse(q, k, lse, 0.25, (action_start, spatial, spatial))
    probabilities = logits.softmax(-1).mean(1)
    weights = torch.tensor([1, 2, 2, 1]) / 6
    reference = torch.stack(
        [
            (
                probabilities[
                    4 * frame : 4 * frame + 4, und + (frame + 1) * spatial : und + (frame + 2) * spatial
                ]
                * weights[:, None]
            ).sum(0)
            for frame in range(8)
        ]
    )
    torch.testing.assert_close(profile, reference, rtol=2e-6, atol=1e-7)


def test_pilot_rejects_wrong_lse_orientation():
    with pytest.raises(ValueError, match="LSE"):
        profile_from_main_lse(
            torch.zeros(100, 4, 8), torch.zeros(100, 2, 8), torch.zeros(1, 4, 32), 0.25, (0, 32, 4)
        )
