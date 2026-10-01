"""Run the original pinned SequencePack code with only its logger import isolated."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sparse_wam.packing import NativePacking, selected_positions


def load_native_helpers(tier):
    path = (
        Path(__file__).parents[1]
        / f"vendor/{tier}/cosmos_framework/data/generator/sequence_packing/runtime.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "cosmos_framework.utils")
    ]
    namespace = {"log": SimpleNamespace(info=lambda *a, **kw: None)}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("tier,spatial", [("edge", 340), ("nano", 360)])
def test_native_metadata_keeps_cfg_lengths_and_original_rope(tier, spatial):
    runtime = load_native_helpers(tier)
    packing = NativePacking.__new__(NativePacking)
    packing.runtime, packing.metadata = runtime, {}
    layout = {
        "latent_positions": {f"L{i}": list(range(i * spatial, (i + 1) * spatial)) for i in range(9)},
        "action_positions": list(range(9 * spatial, 9 * spatial + 33)),
    }
    mask = torch.zeros(8, spatial, dtype=torch.bool)
    for f in range(8):
        mask[f, f : f + 184] = True
    selected = selected_positions(layout, mask)
    for und in (19, 158, 19):
        n = 9 * spatial + 33
        meta = runtime.init_sequence_pack([und + n], [und, n], ["causal", "full"], torch.device("cpu"))
        x = torch.arange(n * 2).reshape(n, 2).float()
        pack = {
            **meta,
            "causal_seq": torch.zeros(und, 2),
            "full_only_seq": x,
            "max_num_tokens": und + n,
            "is_sharded": False,
        }
        rope = ({**pack, "full_only_seq": x + 1000}, {**pack, "full_only_seq": x - 1000})
        packed, sparse_rope = packing.slice(pack, rope, selected)
        assert packed["_num_causal_tokens"] == und
        assert packed["_num_full_tokens"] == spatial + 8 * 184 + 33
        assert packed["sample_offsets"].tolist() == [0, und + len(selected)]
        torch.testing.assert_close(packing.gen(sparse_rope[0]), x[selected] + 1000)
        restored = packing.replace_gen(pack, x.index_copy(0, selected, packing.gen(packed) + 1))
        assert restored["_num_full_tokens"] == n
        assert torch.equal(packing.gen(restored)[selected], x[selected] + 1)
    assert len(packing.metadata) == 2  # CFG branches have independent real UND lengths
