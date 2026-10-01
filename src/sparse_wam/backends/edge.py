# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Use the pinned Edge decoder's main-LSE return path (original B6 plumbing)."""


class EdgeBackend:
    hook_name = "_robolab_version1_controller"

    def __init__(self, config):
        self.config = config
        self.previous = []

    def enter(self, layers):
        for layer in layers:
            attention = layer.self_attn
            if getattr(attention, "_attention_stats_capture_callback", None) is not None:
                raise RuntimeError("Sparse-WAM conflicts with an existing attention capture")
        for layer in layers:
            attention = layer.self_attn
            self.previous.append(
                (
                    attention,
                    hasattr(attention, "_asi_reuse_main_lse"),
                    getattr(attention, "_asi_reuse_main_lse", None),
                )
            )
            attention._asi_reuse_main_lse = True

    def exit(self):
        for attention, existed, value in reversed(self.previous):
            if existed:
                attention._asi_reuse_main_lse = value
            else:
                delattr(attention, "_asi_reuse_main_lse")
        self.previous.clear()

    def run(self, layer, hidden, mask, rope, memory, gen_only, geometry, profile):
        kwargs = {"asi_profile_geometry": geometry} if profile else {}
        output, metadata, kv = layer(
            hidden,
            mask,
            rope,
            natten_metadata=None,
            memory_value=memory,
            gen_only=gen_only,
            **kwargs,
        )
        scores = metadata.pop("asi_profile") if profile else None
        return output, metadata, kv, scores
