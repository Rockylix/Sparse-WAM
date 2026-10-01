# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native Cosmos SequencePack bridge with request-local metadata reuse."""

import torch


def selected_positions(layout, mask):
    device = mask.device
    parts = [torch.tensor(layout["latent_positions"]["L0"], device=device, dtype=torch.long)]
    for frame in range(8):
        positions = torch.tensor(layout["latent_positions"][f"L{frame + 1}"], device=device)
        parts.append(positions[mask[frame]])
    parts.append(torch.tensor(layout["action_positions"], device=device, dtype=torch.long))
    return torch.cat(parts).sort().values


class NativePacking:
    def __init__(self):
        from cosmos_framework.data.generator.sequence_packing import runtime

        self.runtime = runtime
        self.metadata = {}

    def gen(self, pack):
        return self.runtime.get_gen_seq(pack)

    def und(self, pack):
        return self.runtime.get_und_seq(pack)

    def replace_gen(self, pack, gen):
        return self.runtime.from_und_gen_splits(self.und(pack), gen, pack)

    def slice(self, pack, rope, positions):
        und_len = int(pack["_num_causal_tokens"])
        gen = self.gen(pack).index_select(0, positions)
        key = (und_len, gen.shape[0], gen.device)
        if key not in self.metadata:
            self.metadata[key] = self.runtime.init_sequence_pack(
                sample_lens=[und_len + gen.shape[0]],
                split_lens=[und_len, gen.shape[0]],
                attn_modes=["causal", "full"],
                device=gen.device,
            )
        sparse = {
            **self.metadata[key],
            "max_num_tokens": und_len + gen.shape[0],
            "causal_seq": self.und(pack),
            "full_only_seq": gen,
            "is_sharded": False,
        }
        sparse_rope = tuple(
            {
                **sparse,
                "causal_seq": self.und(component),
                "full_only_seq": self.gen(component).index_select(0, positions),
            }
            for component in rope
        )
        return sparse, sparse_rope
