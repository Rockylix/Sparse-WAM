# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# GEN layer execution adapted from version/asi:piper_cosmos/cosmos3/nano_cache_strategies.py.
"""Nano GEN-resident attention; Reasoner prefill/offload remains native."""

from functools import partial

import torch

from ..scoring import profile_from_main_lse


def nano_layer(layer, x, cos, sin, und_k, und_v, geometry, *, profile, attention, run_mlp):
    a = layer.self_attn
    normalized = layer.input_layernorm_moe_gen(x)
    q = a.q_norm_moe_gen(a.q_proj_moe_gen(normalized).view(-1, a.num_attention_heads, a.head_dim))
    k = a.k_norm_moe_gen(a.k_proj_moe_gen(normalized).view(-1, a.num_key_value_heads, a.head_dim))
    v = a.v_proj_moe_gen(normalized).view(-1, a.num_key_value_heads, a.head_dim)
    q, k = a._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    kwargs = dict(
        query=q[None],
        key=torch.cat((und_k, k[None]), 1),
        value=torch.cat((und_v, v[None]), 1),
        scale=a.scaling,
        is_causal=False,
    )
    # Exactly one attention invocation. LSE comes from the same call as the main output.
    if profile:
        output, lse = attention(**kwargs, return_lse=True)
        if lse.ndim == 4 and lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        if tuple(lse.shape) != (1, q.shape[0], q.shape[1]):
            raise RuntimeError("Unexpected main-attention LSE layout")
        start = geometry[0]
        scores = profile_from_main_lse(q, k, lse[:, start : start + 32], a.scaling, geometry)
    else:
        output = attention(**kwargs)
        scores = x.new_empty((0,), dtype=torch.float32)
    z = x + a.o_proj_moe_gen(output[0].flatten(-2, -1))
    residual, _ = run_mlp(layer.mlp_moe_gen, layer.post_attention_layernorm_moe_gen(z))
    return z + residual, scores


class NanoBackend:
    hook_name = "_k112_sparse_controller"  # Native ABI name; no K112 algorithm is used.

    def __init__(self, config):
        self.config = config
        self.cache = None

    def enter(self, layers):
        from cosmos_framework.model.attention import attention
        from cosmos_framework.model.generator.mot.unified_mot import _run_mlp
        from cosmos_framework.data.generator.sequence_packing import runtime

        self.attention, self.run_mlp, self.packing = attention, _run_mlp, runtime
        self.layers = layers
        # Bound to the model layers, so compiled callables survive chunks but never cross models.
        self.cache = getattr(layers[0], "_sparse_wam_nano_kernels", None)
        if self.cache is None:
            self.cache = {}
            layers[0]._sparse_wam_nano_kernels = self.cache
        for layer in layers:
            if getattr(layer.self_attn, "_k112_attention_capture_callback", None) is not None:
                raise RuntimeError("An existing Nano attention capture is active")

    def exit(self):
        pass

    def run(self, layer, hidden, mask, rope, memory, gen_only, geometry, profile):
        if not gen_only or memory is None or memory.und_k is None or memory.und_v is None:
            raise RuntimeError("Nano requires native Reasoner prefill and GEN-only cached UND K/V")
        if (
            getattr(mask, "is_three_way", False)
            or getattr(mask, "control_stream_token_ranges", None) is not None
        ):
            raise ValueError("Only batch-one, full GEN policy attention is supported")
        x = self.packing.get_gen_seq(hidden)
        if x.shape[0] != hidden["_num_full_tokens"]:
            raise ValueError("Nano adapter requires unpadded GEN rows")
        key = (id(layer), profile, self.config.execution)
        if key not in self.cache:
            fn = partial(nano_layer, layer, profile=profile, attention=self.attention, run_mlp=self.run_mlp)
            if self.config.execution != "eager":
                fn = torch.compile(
                    fn,
                    fullgraph=True,
                    dynamic=False,
                    mode="reduce-overhead" if self.config.execution == "compile-graph" else "default",
                )
            self.cache[key] = fn
        cos, sin = (self.packing.get_gen_seq(component) for component in rope)
        y, scores = self.cache[key](x, cos, sin, memory.und_k, memory.und_v, geometry)
        result = self.packing.from_und_gen_splits(self.packing.get_und_seq(hidden), y, hidden)
        return result, {}, None, scores if profile else None
