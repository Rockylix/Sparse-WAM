# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Derived from the pinned Edge C80/S104 and Nano ASI selectors; see SOURCE_MANIFEST.json.
"""Action-guided token selection, paper equations (2)-(6)."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SelectionPlan:
    core: torch.Tensor  # [8, spatial], frame-specific
    stable: torch.Tensor  # [spatial], shared positions, NOT shared features
    keep: torch.Tensor  # [8, spatial]
    layers: torch.Tensor  # six informative layer indices
    quality: torch.Tensor


def select_tokens(profiles: torch.Tensor) -> SelectionPlan:
    """Select exactly Core80 + Stable104 from all layers' raw attention mass.

    The caller supplies layers in original order. No per-frame renormalization
    may be applied before this function: absolute attention mass is significant.
    """
    if profiles.ndim != 3 or profiles.shape[0] not in (28, 36) or profiles.shape[1] != 8:
        raise ValueError("Expected [28 or 36 layers, 8 future frames, spatial]")
    spatial = profiles.shape[-1]
    if spatial < 184:
        raise ValueError("The spatial grid cannot fit Core80 + Stable104")
    if not torch.isfinite(profiles).all() or (profiles < 0).any():
        raise ValueError("Attention profiles must be finite and nonnegative")
    raw = profiles.float()
    eps = torch.finfo(raw.dtype).eps
    mass = raw.sum(-1)
    probability = raw / mass.unsqueeze(-1).clamp_min(eps)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(-1) / math.log(spatial)
    entropy = torch.where(mass > eps, entropy, torch.ones_like(entropy))
    quality = mass.mean(-1) * (1 - entropy.mean(-1)).clamp_min(0)

    layers = quality.topk(6).indices
    core_quality = quality.index_select(0, layers)
    # Degenerate zero-mass input has no ranking signal; retain finite, exact budgets.
    core_weights = torch.where(
        core_quality.sum() > eps,
        core_quality / core_quality.sum().clamp_min(eps),
        torch.full_like(core_quality, 1 / 6),
    )
    core_scores = (raw.index_select(0, layers) * core_weights[:, None, None]).sum(0)
    global_weights = torch.where(quality.sum() > eps, quality, torch.ones_like(quality))
    global_scores = (raw * (global_weights / global_weights.sum())[:, None, None]).sum(0)
    mean = global_scores.mean(0)
    cv = global_scores.std(0, unbiased=False) / mean.clamp_min(eps)
    stable_scores = mean / (1 + cv)

    # Equations (4)-(5): reserve >=104 coordinates outside the union of all cores.
    pool = core_scores.amax(0).topk(spatial - 104).indices
    local = core_scores.index_select(1, pool).topk(80, dim=-1).indices
    core = torch.zeros_like(core_scores, dtype=torch.bool)
    core.scatter_(1, pool[local], True)
    candidates = stable_scores.masked_fill(core.any(0), -torch.inf).topk(104).indices
    stable = torch.zeros(spatial, dtype=torch.bool, device=raw.device)
    stable[candidates] = True
    keep = core | stable[None]
    if not (keep.sum(-1) == 184).all() or (core & stable).any():
        raise RuntimeError("Core/Stable selection violated its disjoint budget")
    return SelectionPlan(core, stable, keep, layers, quality)
