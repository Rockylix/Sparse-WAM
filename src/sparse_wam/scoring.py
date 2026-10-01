# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pilot: local QK products normalized by the MAIN attention's full-key LSE."""

import torch


def profile_from_main_lse(q_gen, k_gen, action_lse, scaling, geometry):
    """Compute [8, spatial] scores without a second attention or value output.

    Q/K are post-RoPE [tokens, heads, dim]. LSE is natural-log, float32,
    [1, 32, query_heads], and MUST include all actually visible UND + GEN keys.
    ``geometry`` is (first predicted action, first future token, spatial).
    """
    action_start, future_start, spatial = geometry
    heads, width = q_gen.shape[1:]
    kv_heads = k_gen.shape[1]
    if heads % kv_heads or tuple(action_lse.shape) != (1, 32, heads):
        raise ValueError("Invalid GQA geometry or main-attention LSE shape")
    queries = q_gen[action_start : action_start + 32].float()
    queries = queries.reshape(8, 4, heads, width).permute(0, 2, 1, 3).contiguous()
    keys = k_gen[future_start : future_start + 8 * spatial].float()
    keys = keys.repeat_interleave(heads // kv_heads, dim=1)
    keys = keys.reshape(8, spatial, heads, width).permute(0, 2, 3, 1).contiguous()
    logits = torch.matmul(queries * scaling, keys)
    normalizer = action_lse.float().reshape(8, 4, heads).permute(0, 2, 1).unsqueeze(-1)
    probabilities = torch.exp(logits - normalizer).mean(dim=1)
    weights = probabilities.new_tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    return (probabilities * weights[None, :, None]).sum(dim=1)


def profile_geometry(layout):
    actions = [q["gen_position"] for q in layout["action_queries"] if q["query_role"] == "predicted"]
    future = [p for f in range(1, 9) for p in layout["latent_positions"][f"L{f}"]]
    spatial = len(layout["latent_positions"]["L1"])
    if len(actions) != 32 or actions != list(range(actions[0], actions[0] + 32)):
        raise ValueError("Expected 32 contiguous predicted action positions")
    if future != list(range(future[0], future[0] + 8 * spatial)):
        raise ValueError("Expected contiguous, frame-major future positions")
    return actions[0], future[0], spatial
