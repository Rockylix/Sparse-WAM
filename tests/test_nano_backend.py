"""Numerical reference for the actual Nano GEN layer and its single-attention scorer."""

from types import SimpleNamespace

import torch

from sparse_wam.backends.nano import nano_layer


def test_nano_main_attention_output_matches_dense_and_scores_without_second_attention():
    torch.manual_seed(18)
    hidden, heads, kv_heads, width, spatial = 8, 4, 2, 2, 3
    n = 9 * spatial + 33
    action_start = 9 * spatial + 1
    a = SimpleNamespace(
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=width,
        scaling=width**-0.5,
        q_proj_moe_gen=torch.nn.Linear(hidden, heads * width, bias=False),
        k_proj_moe_gen=torch.nn.Linear(hidden, kv_heads * width, bias=False),
        v_proj_moe_gen=torch.nn.Linear(hidden, kv_heads * width, bias=False),
        q_norm_moe_gen=torch.nn.Identity(),
        k_norm_moe_gen=torch.nn.Identity(),
        o_proj_moe_gen=torch.nn.Linear(heads * width, hidden, bias=False),
        _apply_rotary_pos_emb=lambda q, k, cos, sin, **kwargs: (q, k),
    )
    layer = SimpleNamespace(
        self_attn=a,
        input_layernorm_moe_gen=torch.nn.LayerNorm(hidden),
        post_attention_layernorm_moe_gen=torch.nn.LayerNorm(hidden),
        mlp_moe_gen=torch.nn.Sequential(
            torch.nn.Linear(hidden, 12), torch.nn.SiLU(), torch.nn.Linear(12, hidden)
        ),
    )
    x = torch.randn(n, hidden)
    ku, vu = torch.randn(1, 7, kv_heads, width), torch.randn(1, 7, kv_heads, width)
    calls = []

    def reference_attention(*, query, key, value, scale, is_causal, return_lse=False):
        calls.append(return_lse)
        k, v = (item.repeat_interleave(heads // kv_heads, 2) for item in (key, value))
        logits = torch.einsum("bqhd,bkhd->bqhk", query * scale, k)
        output = torch.einsum("bqhk,bkhd->bqhd", logits.softmax(-1), v)
        return (output, logits.logsumexp(-1)) if return_lse else output

    kwargs = dict(attention=reference_attention, run_mlp=lambda mlp, inp: (mlp(inp), None))
    with torch.no_grad():
        dense, _ = nano_layer(
            layer, x, None, None, ku, vu, (action_start, spatial, spatial), profile=False, **kwargs
        )
        profiled, scores = nano_layer(
            layer, x, None, None, ku, vu, (action_start, spatial, spatial), profile=True, **kwargs
        )
    assert calls == [False, True]  # ONE main attention per layer, no second action-row call
    torch.testing.assert_close(dense, profiled, rtol=0, atol=0)
    normalized = layer.input_layernorm_moe_gen(x)
    q = a.q_proj_moe_gen(normalized).reshape(n, heads, width)
    k = a.k_proj_moe_gen(normalized).reshape(n, kv_heads, width)
    keys = torch.cat((ku[0], k)).repeat_interleave(heads // kv_heads, 1)
    probabilities = torch.einsum("qhd,khd->qhk", q[action_start:] * a.scaling, keys).softmax(-1).mean(1)
    expected = torch.stack(
        [
            (
                probabilities[4 * f : 4 * f + 4, 7 + (f + 1) * spatial : 7 + (f + 2) * spatial]
                * torch.tensor([1, 2, 2, 1])[:, None]
                / 6
            ).sum(0)
            for f in range(8)
        ]
    )
    torch.testing.assert_close(scores, expected, rtol=2e-6, atol=1e-7)
