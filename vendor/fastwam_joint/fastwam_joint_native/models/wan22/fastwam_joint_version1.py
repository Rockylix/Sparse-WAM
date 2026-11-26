from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .wan_video_dit import sinusoidal_embedding_1d
from .fastwam_step0_profile import (
    ACTION_WEIGHTING,
    ACTION_WEIGHTS,
    aligned_action_future_profile_from_lse,
    weighted_aligned_action_future_profile,
)


EPS = 1e-12
STRATEGY_FAMILY = "fastwam-joint-version1-global-b0-b29-core-stable"


def _topk_mask(scores: torch.Tensor, count: int, *, allowed: torch.Tensor | None = None) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected one-dimensional scores, got {tuple(scores.shape)}")
    candidates = (
        torch.arange(scores.numel(), device=scores.device)
        if allowed is None
        else torch.nonzero(allowed, as_tuple=False).flatten()
    )
    if count < 0 or count > int(candidates.numel()):
        raise ValueError(f"Cannot select {count} tokens from {int(candidates.numel())} candidates")
    selected = candidates.index_select(0, torch.topk(scores.index_select(0, candidates), count).indices)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[selected] = True
    return mask


@dataclass(frozen=True)
class FastWAMJointVersion1Plan:
    raw_profiles: torch.Tensor
    spatial_profiles: torch.Tensor
    block_mass: torch.Tensor
    block_entropy: torch.Tensor
    block_quality: torch.Tensor
    selected_blocks_ranked: torch.Tensor
    selected_block_weights: torch.Tensor
    core_scores: torch.Tensor
    core_masks: torch.Tensor
    global_latent_scores: torch.Tensor
    stable_score: torch.Tensor
    stable_mask: torch.Tensor
    fixed_masks: torch.Tensor
    attention_mass_retention: torch.Tensor


@torch.no_grad()
def build_fastwam_joint_version1_plan(
    raw_profiles: torch.Tensor,
    *,
    top_block_count: int = 6,
    core_token_count: int = 19,
    stable_token_count: int = 37,
    stable_cv_penalty: float = 1.0,
    balanced_budget: bool = False,
) -> FastWAMJointVersion1Plan:
    """Build the global B0-B29 fixed Core+Stable execution plan.

    ``raw_profiles`` is the complete-softmax Action-aligned probability mass
    with shape ``[block, future_latent, spatial_token]``. Selection follows the
    current Edge Version1 raw-mass contract; spatial normalization is retained
    separately for entropy and visualization.
    """

    if raw_profiles.ndim != 3:
        raise ValueError(f"Expected [block,future,spatial], got {tuple(raw_profiles.shape)}")
    blocks, future_latents, spatial = (int(value) for value in raw_profiles.shape)
    if blocks != 30 or future_latents != 2 or spatial != 98:
        raise ValueError(
            "FastWAM Joint Version1 expects [30,2,98] profiles, "
            f"got {tuple(raw_profiles.shape)}"
        )
    if top_block_count <= 0 or top_block_count > blocks:
        raise ValueError(f"Invalid top_block_count={top_block_count}")
    if min(core_token_count, stable_token_count) < 0 or (not balanced_budget and min(core_token_count, stable_token_count) == 0):
        raise ValueError("Core and Stable token counts must be positive")
    if core_token_count + stable_token_count > spatial:
        raise ValueError("Core+Stable token budget exceeds one future latent")
    if stable_cv_penalty < 0:
        raise ValueError("stable_cv_penalty must be non-negative")
    if not bool(torch.isfinite(raw_profiles).all()) or bool((raw_profiles < 0).any()):
        raise ValueError("Raw attention profiles must be finite and non-negative")

    raw = raw_profiles.float()
    frame_mass = raw.sum(dim=-1)
    spatial_probability = raw / frame_mass.unsqueeze(-1).clamp_min(EPS)
    entropy = -(spatial_probability * spatial_probability.clamp_min(EPS).log()).sum(dim=-1)
    entropy = entropy / math.log(spatial)
    entropy = torch.where(frame_mass > EPS, entropy, torch.ones_like(entropy))
    block_mass = frame_mass.mean(dim=1)
    block_entropy = entropy.mean(dim=1)
    block_quality = block_mass * (1.0 - block_entropy)

    selected_blocks = torch.topk(block_quality, top_block_count).indices
    selected_quality = block_quality.index_select(0, selected_blocks)
    if float(selected_quality.sum()) <= EPS:
        selected_quality = torch.ones_like(selected_quality)
    selected_weights = selected_quality / selected_quality.sum()
    core_scores = (
        raw.index_select(0, selected_blocks) * selected_weights[:, None, None]
    ).sum(dim=0)

    global_weights = block_quality.clamp_min(0)
    if float(global_weights.sum()) <= EPS:
        global_weights = torch.ones_like(global_weights)
    global_weights = global_weights / global_weights.sum()
    global_latent_scores = (raw * global_weights[:, None, None]).sum(dim=0)
    stable_mean = global_latent_scores.mean(dim=0)
    stable_std = global_latent_scores.std(dim=0, unbiased=False)
    stable_cv = stable_std / stable_mean.clamp_min(EPS)
    stable_score = stable_mean / (1.0 + float(stable_cv_penalty) * stable_cv)
    if balanced_budget:
        # Opt-in scaling experiment: reserve shared Stable first, then select
        # disjoint per-future Core. This admits equal budgets all the way to 98.
        stable_mask = _topk_mask(stable_score, stable_token_count)
        core_masks = torch.stack([_topk_mask(core_scores[latent], core_token_count, allowed=~stable_mask) for latent in range(2)])
    else:
        core_masks = torch.stack([_topk_mask(core_scores[latent], core_token_count) for latent in range(2)])
        core_union = core_masks.any(dim=0)
        stable_mask = _topk_mask(stable_score, stable_token_count, allowed=~core_union)
    fixed_masks = core_masks | stable_mask.unsqueeze(0)

    expected = core_token_count + stable_token_count
    if not bool((core_masks.sum(dim=-1) == core_token_count).all()):
        raise RuntimeError("Core mask count changed")
    if int(stable_mask.sum()) != stable_token_count:
        raise RuntimeError("Stable mask count changed")
    if bool((core_masks & stable_mask.unsqueeze(0)).any()):
        raise RuntimeError("Core and Stable masks overlap")
    if not bool((fixed_masks.sum(dim=-1) == expected).all()):
        raise RuntimeError("Fixed execution mask count changed")

    retained = (raw * fixed_masks.unsqueeze(0)).sum(dim=-1)
    retention = retained / frame_mass.clamp_min(EPS)
    return FastWAMJointVersion1Plan(
        raw_profiles=raw,
        spatial_profiles=spatial_probability,
        block_mass=block_mass,
        block_entropy=block_entropy,
        block_quality=block_quality,
        selected_blocks_ranked=selected_blocks,
        selected_block_weights=selected_weights,
        core_scores=core_scores,
        core_masks=core_masks,
        global_latent_scores=global_latent_scores,
        stable_score=stable_score,
        stable_mask=stable_mask,
        fixed_masks=fixed_masks,
        attention_mass_retention=retention,
    )


@torch.no_grad()
def apply_random_fastwam_joint_version1_selection(
    plan: FastWAMJointVersion1Plan,
    *,
    token_count: int,
    seed: int = 7301,
) -> FastWAMJointVersion1Plan:
    """Replace Core selection with uniform per-future random masks.

    A private CPU generator keeps mask generation deterministic without
    consuming the policy's CPU or CUDA sampling RNG streams.
    """
    if token_count < 0 or token_count > int(plan.raw_profiles.shape[-1]):
        raise ValueError(f"Invalid random token_count={token_count}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    masks = []
    for _ in range(int(plan.raw_profiles.shape[1])):
        indices = torch.randperm(int(plan.raw_profiles.shape[-1]), generator=generator)[:token_count]
        mask = torch.zeros(int(plan.raw_profiles.shape[-1]), dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    core_masks = torch.stack(masks).to(device=plan.raw_profiles.device)
    stable_mask = torch.zeros_like(plan.stable_mask)
    fixed_masks = core_masks | stable_mask.unsqueeze(0)
    raw = plan.raw_profiles.float()
    frame_mass = raw.sum(dim=-1)
    retained = (raw * fixed_masks.unsqueeze(0)).sum(dim=-1)
    retention = retained / frame_mass.clamp_min(EPS)
    return replace(
        plan,
        core_masks=core_masks,
        stable_mask=stable_mask,
        fixed_masks=fixed_masks,
        attention_mass_retention=retention,
    )


class FastWAMJointVersion1Controller:
    """One-request controller for global-profile fixed-mask sparse inference."""

    def __init__(
        self,
        *,
        mot: Any,
        num_steps: int,
        top_block_count: int = 6,
        core_token_count: int = 19,
        stable_token_count: int = 37,
        stable_cv_penalty: float = 1.0,
        velocity_cache: bool = True,
        torch_compile: bool = True,
        cuda_graph: bool = True,
        balanced_budget: bool = False,
        selection_strategy: str = "profile",
        random_selection_seed: int = 7301,
        dense_steps: Sequence[int] = (0,),
    ) -> None:
        self.mot = mot
        self.num_steps = int(num_steps)
        self.dense_steps = tuple(int(step) for step in dense_steps)
        if (not self.dense_steps or self.dense_steps[0] != 0
                or tuple(sorted(set(self.dense_steps))) != self.dense_steps
                or self.dense_steps[-1] >= self.num_steps):
            raise ValueError("dense_steps must be sorted, unique, start at 0 and lie within num_steps")
        self._dense_steps_seen: list[int] = []
        self._graph_invariant_refresh_steps: list[int] = []
        self._background_cache_source_step: int | None = None
        self.top_block_count = int(top_block_count)
        self.core_token_count = int(core_token_count)
        self.stable_token_count = int(stable_token_count)
        self.stable_cv_penalty = float(stable_cv_penalty)
        self.velocity_cache = bool(velocity_cache)
        self.torch_compile = bool(torch_compile)
        self.cuda_graph = bool(cuda_graph)
        self.balanced_budget = bool(balanced_budget)
        if selection_strategy not in ("profile", "random"):
            raise ValueError(f"Unsupported selection_strategy={selection_strategy!r}")
        self.selection_strategy = selection_strategy
        self.random_selection_seed = int(random_selection_seed)
        self.torch_compile_fallback_error: str | None = None
        self.graph_fallback_error: str | None = None
        if int(getattr(mot, "num_layers", -1)) != 30:
            raise ValueError("FastWAM Joint Version1 requires exactly 30 MoT blocks")
        video_expert = getattr(getattr(mot, "mixtures", {}), "get", lambda _name: None)("video")
        if video_expert is None and hasattr(mot, "mixtures"):
            try:
                video_expert = mot.mixtures["video"]
            except (KeyError, TypeError):
                video_expert = None
        attention_mode = getattr(video_expert, "video_attention_mask_mode", None)
        if video_expert is not None and attention_mode != "first_frame_causal":
            raise ValueError(
                "Version1 condition K/V reuse requires video_attention_mask_mode='first_frame_causal', "
                f"got {attention_mode!r}"
            )
        if self.num_steps < 2:
            raise ValueError("Version1 requires at least two denoise steps")

        self.current_step: int | None = None
        self.video_grid: tuple[int, int, int] | None = None
        self.plan: FastWAMJointVersion1Plan | None = None
        self._step0_profiles: dict[int, torch.Tensor] = {}
        self._step0_profile_buffer: torch.Tensor | None = None
        self._step0_profile_partial_workspace: torch.Tensor | None = None
        self._step0_attention_bias_storage: torch.Tensor | None = None
        self._step0_attention_bias: torch.Tensor | None = None
        self._fused_step0_profile_enabled = True
        self._fused_step0_profile_used = False
        self._fused_step0_profile_error: str | None = None
        self._condition_kv_by_block: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._full_video_side_buffer: torch.Tensor | None = None
        self._video_keep_positions: torch.Tensor | None = None
        self._fixed_video_keep_positions: torch.Tensor | None = None
        self._fixed_joint_query_positions: torch.Tensor | None = None
        self._fixed_joint_key_value_positions: torch.Tensor | None = None
        self._step0_video_velocity: torch.Tensor | None = None
        self._step0_attention_mask: torch.Tensor | None = None
        self._step0_freqs_all: dict[str, torch.Tensor] | None = None
        self._step0_context_all: dict[str, Mapping[str, torch.Tensor] | None] | None = None
        self._fixed_sparse_attention_mask: torch.Tensor | None = None
        self._persistent_freqs_all: dict[str, torch.Tensor] | None = None
        self._persistent_context_all: dict[str, Mapping[str, torch.Tensor] | None] | None = None
        self._cross_attention_kv_by_block: tuple[
            dict[str, tuple[torch.Tensor, torch.Tensor]], ...
        ] | None = None
        self._step0_cross_attention_kv_by_block: dict[
            int, dict[str, tuple[torch.Tensor, torch.Tensor]]
        ] = {}
        self._cross_attention_kv_reused_from_step0 = False
        self.persistent_action_only = False
        self._persistent_pack_count = 0
        self._persistent_full_video_state: torch.Tensor | None = None
        self._background_cache_update_steps: list[int] = []
        self._completed_steps: list[int] = []
        self._velocity_replaced_elements = 0
        self._sparse_kernels_prepared = False
        self._kernel_storage_validation_steps: list[int] = []
        self._attached = False

    def __enter__(self) -> "FastWAMJointVersion1Controller":
        if getattr(self.mot, "version1_sparse_controller", None) is not None:
            raise RuntimeError("MoT already has a Version1 sparse controller")
        if getattr(self.mot, "action_future_attention_observer", None) is not None:
            raise RuntimeError("Version1 sparse inference cannot share the read-only attention observer")
        self.mot.version1_sparse_controller = self
        self._attached = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        if self._attached:
            self.mot.version1_sparse_controller = None
            self._attached = False
        if exc_type is None:
            expected = list(range(self.num_steps))
            if self._completed_steps != expected:
                raise RuntimeError(
                    f"Version1 observed denoise steps {self._completed_steps}, expected {expected}"
                )
            if self.plan is None:
                raise RuntimeError("Version1 request finished without a mask plan")
            if self.persistent_action_only and self.velocity_cache:
                if self._background_cache_update_steps != [s for s in range(self.num_steps) if s not in self.dense_steps]:
                    raise RuntimeError("Version1 request finished without updating every background cache step")

    def is_dense_step(self, step: int | None = None) -> bool:
        return (self.current_step if step is None else step) in self.dense_steps

    def full_video_state_for_refresh(self) -> torch.Tensor:
        if not self.velocity_cache or self._persistent_full_video_state is None:
            raise RuntimeError("Dense refresh requires the current full latent background cache")
        return self._persistent_full_video_state

    def enable_persistent_action_only(self) -> None:
        """Compute selected patches and optionally maintain a full latent cache."""
        self.persistent_action_only = True

    def prepare_forward(
        self,
        *,
        step_index: int | None,
        video_meta: Mapping[str, Any],
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> None:
        del timestep_video, timestep_action
        if step_index is None:
            raise ValueError("Version1 requires an explicit denoise step index")
        step = int(step_index)
        if step < 0 or step >= self.num_steps:
            raise ValueError(f"Invalid Version1 denoise step {step}/{self.num_steps}")
        if step == 0:
            # Prompt encoding may offload MoT CPU->GPU after controller entry.
            # Prepare/rebuild shared QKV storage only once the denoise device is final.
            self.mot.prepare_version1_sparse_kernels(allow_rebuild=True)
            self._sparse_kernels_prepared = True
            self._kernel_storage_validation_steps = [0]
            self.plan = None
            self._step0_profiles.clear()
            self._condition_kv_by_block.clear()
            self._step0_video_velocity = None
            self._step0_attention_mask = None
            self._step0_freqs_all = None
            self._step0_context_all = None
            self._fixed_sparse_attention_mask = None
            self._persistent_freqs_all = None
            self._persistent_context_all = None
            self._cross_attention_kv_by_block = None
            self._step0_cross_attention_kv_by_block.clear()
            self._cross_attention_kv_reused_from_step0 = False
            self._persistent_pack_count = 0
            self._persistent_full_video_state = None
            self._background_cache_update_steps.clear()
            self._fixed_video_keep_positions = None
            self._fixed_joint_query_positions = None
            self._fixed_joint_key_value_positions = None
            self._completed_steps.clear()
            self._velocity_replaced_elements = 0
            self._dense_steps_seen.clear()
            self._graph_invariant_refresh_steps.clear()
        elif self.is_dense_step(step):
            # Preserve the fully evolved background and request counters. Dense
            # forward replaces all per-refresh profiles, K/V and sparse workspace.
            self.plan = None
            self._step0_profiles.clear()
            self._condition_kv_by_block.clear()
            self._step0_cross_attention_kv_by_block.clear()
            self._cross_attention_kv_by_block = None
            self._cross_attention_kv_reused_from_step0 = False
            self._step0_video_velocity = None
        elif self.plan is None:
            raise RuntimeError("Sparse denoise step started before the global Step-0 profile was finalized")
        elif sorted(self._condition_kv_by_block) != list(range(30)):
            raise RuntimeError("Sparse denoise step started without complete B0-B29 condition K/V cache")
        expected_step = len(self._completed_steps)
        if step != expected_step:
            raise RuntimeError(f"Version1 denoise step order changed: got {step}, expected {expected_step}")

        grid = tuple(int(value) for value in video_meta["grid_size"])
        if grid != (3, 7, 14):
            raise ValueError(f"Version1 expects FastWAM video grid (3,7,14), got {grid}")
        self.video_grid = grid
        self.current_step = step
        if self.is_dense_step():
            self._dense_steps_seen.append(step)

    @torch.no_grad()
    def prepare_mot_inputs(
        self,
        *,
        embeds_all: Mapping[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Mapping[str, torch.Tensor],
        context_all: Mapping[str, Mapping[str, torch.Tensor] | None],
        t_mod_all: Mapping[str, torch.Tensor],
        expert_order: Sequence[str],
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, Mapping[str, torch.Tensor] | None],
        dict[str, torch.Tensor],
    ]:
        if self.current_step is None or self.video_grid is None:
            raise RuntimeError("prepare_forward must run before MoT input preparation")
        if list(expert_order) != ["video", "action"]:
            raise ValueError(f"Version1 requires expert order ['video','action'], got {list(expert_order)}")
        if self.is_dense_step():
            self._step0_attention_mask = attention_mask.detach().clone()
            self._step0_freqs_all = {
                name: value.detach().clone() for name, value in freqs_all.items()
            }
            self._step0_context_all = {
                name: (
                    None
                    if payload is None
                    else {
                        key: value.detach().clone()
                        for key, value in payload.items()
                    }
                )
                for name, payload in context_all.items()
            }
            return dict(embeds_all), attention_mask, dict(freqs_all), dict(context_all), dict(t_mod_all)
        if self.plan is None:
            raise RuntimeError("Version1 fixed mask is unavailable")

        video = embeds_all["video"]
        action = embeds_all["action"]
        if tuple(video.shape[:2]) != (1, 294) or tuple(action.shape[:2]) != (1, 32):
            raise ValueError(
                f"Version1 expects video/action token shapes [1,294]/[1,32], got {video.shape}/{action.shape}"
            )
        device = video.device
        if (
            self._fixed_video_keep_positions is None
            or self._fixed_joint_query_positions is None
            or self._fixed_joint_key_value_positions is None
        ):
            raise RuntimeError("Version1 Step0 sparse index workspace is unavailable")
        video_keep = self._fixed_video_keep_positions
        joint_query_keep = self._fixed_joint_query_positions
        joint_key_value_keep = self._fixed_joint_key_value_positions
        if video_keep.device != device:
            raise RuntimeError("Version1 sparse index workspace moved to a different device")
        if int(video_keep.numel()) != 2 * (self.core_token_count + self.stable_token_count):
            raise RuntimeError("Version1 packed video length changed")

        self._full_video_side_buffer = video.detach().clone()
        self._video_keep_positions = video_keep
        packed_embeds = dict(embeds_all)
        packed_embeds["video"] = video.index_select(1, video_keep)

        packed_freqs = dict(freqs_all)
        packed_freqs["video"] = freqs_all["video"].index_select(0, video_keep)

        packed_t_mod = dict(t_mod_all)
        video_t_mod = t_mod_all["video"]
        if video_t_mod.ndim != 4 or int(video_t_mod.shape[1]) != 294:
            raise ValueError(f"Unexpected video t_mod shape {tuple(video_t_mod.shape)}")
        packed_t_mod["video"] = video_t_mod.index_select(1, video_keep)

        packed_context = dict(context_all)
        video_context = context_all["video"]
        if video_context is not None:
            video_context = dict(video_context)
            context_mask = video_context.get("mask")
            if context_mask is not None:
                if context_mask.ndim != 3 or int(context_mask.shape[1]) != 294:
                    raise ValueError(f"Unexpected video context mask shape {tuple(context_mask.shape)}")
                video_context["mask"] = context_mask.index_select(1, video_keep)
            packed_context["video"] = video_context

        packed_attention_mask = attention_mask.index_select(0, joint_query_keep).index_select(
            1, joint_key_value_keep
        )
        return packed_embeds, packed_attention_mask, packed_freqs, packed_context, packed_t_mod

    @torch.no_grad()
    def observe_dense_block(
        self,
        *,
        layer_index: int,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        expert_order: Sequence[str],
        seq_lens: Sequence[int],
        attention_logsumexp: torch.Tensor | None = None,
    ) -> None:
        if not self.is_dense_step():
            return
        block = int(layer_index)
        if block in self._step0_profiles:
            raise RuntimeError(f"Duplicate Version1 Step-0 profile for B{block}")
        if list(expert_order) != ["video", "action"] or [int(x) for x in seq_lens] != [294, 32]:
            raise ValueError("Version1 Step-0 profile requires full video/action sequences")
        if q_cat.shape != k_cat.shape or k_cat.shape != v_cat.shape or tuple(q_cat.shape[:2]) != (1, 326):
            raise ValueError(
                "Unexpected Version1 Q/K/V shape "
                f"{tuple(q_cat.shape)}/{tuple(k_cat.shape)}/{tuple(v_cat.shape)}"
            )
        hidden = int(q_cat.shape[-1])
        if hidden % int(num_heads):
            raise ValueError("Attention hidden size is not divisible by num_heads")
        head_dim = hidden // int(num_heads)

        self._condition_kv_by_block[block] = (
            k_cat[:, :98].detach().clone(),
            v_cat[:, :98].detach().clone(),
        )

        aligned = None
        if attention_logsumexp is not None and self._fused_step0_profile_enabled:
            try:
                if self._step0_profile_buffer is None:
                    self._step0_profile_buffer = torch.empty(
                        (30, 2, 98), dtype=torch.float32, device=q_cat.device
                    )
                    self._step0_profile_partial_workspace = torch.empty(
                        (2, 7, 6, 16), dtype=torch.float32, device=q_cat.device
                    )
                if self._step0_profile_partial_workspace is None:
                    raise RuntimeError("Fused Step0 profile workspace is unavailable")
                aligned = aligned_action_future_profile_from_lse(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    attention_logsumexp=attention_logsumexp,
                    partial_workspace=self._step0_profile_partial_workspace,
                    output=self._step0_profile_buffer[block],
                )
                self._fused_step0_profile_used = True
            except Exception as exc:
                self._fused_step0_profile_enabled = False
                self._fused_step0_profile_error = f"{type(exc).__name__}: {exc}"
                aligned = None
        if aligned is None:
            q_action = q_cat[:, 294:326].reshape(1, 32, int(num_heads), head_dim).permute(0, 2, 1, 3)
            k_all = k_cat.reshape(1, 326, int(num_heads), head_dim).permute(0, 2, 1, 3)
            logits = torch.matmul(q_action.float(), k_all.float().transpose(-1, -2)) / math.sqrt(head_dim)
            allowed = attention_mask[294:326].to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
            probability = torch.softmax(logits, dim=-1)
            future = probability[..., 98:294].reshape(1, int(num_heads), 32, 2, 98)
            head_mean = future.mean(dim=1)[0]
            aligned = weighted_aligned_action_future_profile(head_mean)
        self._step0_profiles[block] = aligned.detach()

    def prepend_condition_kv(
        self,
        *,
        layer_index: int,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore invariant first-frame K/V for rectangular sparse attention."""
        if self.is_dense_step():
            return k_cat, v_cat
        block = int(layer_index)
        cached = self._condition_kv_by_block.get(block)
        if cached is None:
            raise RuntimeError(f"Missing Step-0 condition K/V cache for B{block}")
        condition_k, condition_v = cached
        return torch.cat([condition_k, k_cat], dim=1), torch.cat([condition_v, v_cat], dim=1)

    def condition_kv_by_block(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if sorted(self._condition_kv_by_block) != list(range(30)):
            raise RuntimeError("Incomplete B0-B29 condition K/V cache")
        return tuple(self._condition_kv_by_block[block] for block in range(30))

    def cross_attention_kv_by_block(
        self,
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...]:
        if self._cross_attention_kv_by_block is None:
            raise RuntimeError("Incomplete B0-B29 cross-attention K/V cache")
        return self._cross_attention_kv_by_block

    @torch.no_grad()
    def capture_step0_cross_attention_kv(
        self,
        *,
        layer_index: int,
        expert_name: str,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Retain the context K/V already computed by the dense Step0 block."""
        if not self.is_dense_step():
            return
        layer = int(layer_index)
        name = str(expert_name)
        if layer < 0 or layer >= 30:
            raise ValueError(f"Invalid Version1 cross-attention block {layer}")
        if name not in ("video", "action"):
            raise ValueError(f"Invalid Version1 cross-attention expert {name!r}")
        layer_cache = self._step0_cross_attention_kv_by_block.setdefault(layer, {})
        if name in layer_cache:
            raise RuntimeError(f"Duplicate Version1 Step0 cross-attention K/V for B{layer}/{name}")
        if k.ndim != 3 or v.ndim != 3 or tuple(k.shape[:2]) != tuple(v.shape[:2]):
            raise ValueError(
                f"Invalid Version1 cross-attention K/V shape for B{layer}/{name}: "
                f"{tuple(k.shape)}/{tuple(v.shape)}"
            )
        layer_cache[name] = (k.detach(), v.detach())

    @torch.no_grad()
    def _prepare_cross_attention_kv_cache(self) -> None:
        if self._step0_context_all is None:
            raise RuntimeError("Version1 Step0 context payload is unavailable")
        complete_capture = (
            sorted(self._step0_cross_attention_kv_by_block) == list(range(30))
            and all(
                set(self._step0_cross_attention_kv_by_block[layer]) == {"video", "action"}
                for layer in range(30)
            )
        )
        if complete_capture:
            self._cross_attention_kv_by_block = tuple(
                dict(self._step0_cross_attention_kv_by_block[layer])
                for layer in range(30)
            )
            self._cross_attention_kv_reused_from_step0 = True
            return
        if self._step0_cross_attention_kv_by_block:
            captured = {
                layer: sorted(values)
                for layer, values in self._step0_cross_attention_kv_by_block.items()
            }
            raise RuntimeError(f"Incomplete Version1 Step0 cross-attention K/V capture: {captured}")
        cached_blocks: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
        for layer_index in range(30):
            layer_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for name in ("video", "action"):
                payload = self._step0_context_all.get(name)
                if payload is None or payload.get("context") is None:
                    raise RuntimeError(f"Version1 Step0 {name} context is unavailable")
                block = self.mot.mixtures[name].blocks[layer_index]
                cross_attention = block.cross_attn
                embedded_context = payload["context"]
                layer_cache[name] = (
                    cross_attention.norm_k(cross_attention.k(embedded_context)).detach(),
                    cross_attention.v(embedded_context).detach(),
                )
            cached_blocks.append(layer_cache)
        self._cross_attention_kv_by_block = tuple(cached_blocks)
        self._cross_attention_kv_reused_from_step0 = False

    @torch.no_grad()
    def _prepare_sparse_workspace(self) -> None:
        if self.plan is None:
            raise RuntimeError("Version1 cannot prepare sparse workspace without a fixed mask plan")
        device = self.plan.fixed_masks.device
        future = []
        for latent in range(2):
            local = torch.nonzero(self.plan.fixed_masks[latent], as_tuple=False).flatten()
            future.append(local + (latent + 1) * 98)
        self._fixed_video_keep_positions = torch.cat(future)
        action_keep = torch.arange(32, device=device, dtype=torch.long) + 294
        condition_keep = torch.arange(98, device=device, dtype=torch.long)
        self._fixed_joint_query_positions = torch.cat([self._fixed_video_keep_positions, action_keep])
        self._fixed_joint_key_value_positions = torch.cat(
            [condition_keep, self._fixed_video_keep_positions, action_keep]
        )
        if (
            self._step0_attention_mask is None
            or self._step0_freqs_all is None
            or self._step0_context_all is None
        ):
            # Direct plan/index unit tests do not execute a dense Step0. A real
            # persistent request must populate these invariants before Step1.
            return
        self._fixed_sparse_attention_mask = self._step0_attention_mask.index_select(
            0, self._fixed_joint_query_positions
        ).index_select(1, self._fixed_joint_key_value_positions)
        self._persistent_freqs_all = {
            "video": self._step0_freqs_all["video"].index_select(
                0, self._fixed_video_keep_positions
            ),
            "action": self._step0_freqs_all["action"],
        }
        persistent_context: dict[str, Mapping[str, torch.Tensor] | None] = {}
        for name in ("video", "action"):
            payload = self._step0_context_all[name]
            if payload is None:
                persistent_context[name] = None
                continue
            packed_payload = dict(payload)
            mask = packed_payload.get("mask")
            if name == "video" and mask is not None:
                packed_payload["mask"] = mask.index_select(
                    1, self._fixed_video_keep_positions
                )
            persistent_context[name] = packed_payload
        self._persistent_context_all = persistent_context
        self._prepare_cross_attention_kv_cache()

    @torch.no_grad()
    def pack_persistent_video_state(self, latents_video: torch.Tensor) -> torch.Tensor:
        """Gather selected latent patches once after the dense Step0 scheduler update."""
        if not self.persistent_action_only:
            raise RuntimeError("Persistent video state is only valid for action-only inference")
        if self.plan is None or self._fixed_video_keep_positions is None or self.video_grid is None:
            raise RuntimeError("Version1 sparse plan is unavailable for persistent packing")
        if self._persistent_pack_count >= max(1, len(self._dense_steps_seen)):
            raise RuntimeError("Version1 persistent video state may only be packed once per dense refresh")
        if latents_video.ndim != 5 or int(latents_video.shape[0]) != 1:
            raise ValueError(f"Expected video latents [1,C,T,H,W], got {tuple(latents_video.shape)}")
        patch_t, patch_h, patch_w = (
            int(value) for value in self.mot.mixtures["video"].patch_size
        )
        grid_t, grid_h, grid_w = self.video_grid
        expected = (grid_t * patch_t, grid_h * patch_h, grid_w * patch_w)
        if tuple(int(value) for value in latents_video.shape[2:]) != expected:
            raise ValueError(
                f"Video latent shape {tuple(latents_video.shape[2:])} does not match grid/patch {expected}"
            )
        patches = (
            latents_video.reshape(
                1,
                int(latents_video.shape[1]),
                grid_t,
                patch_t,
                grid_h,
                patch_h,
                grid_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(1, grid_t * grid_h * grid_w, patch_t, patch_h, patch_w, -1)
        )
        packed = patches.index_select(1, self._fixed_video_keep_positions).contiguous()
        if self.velocity_cache:
            self._persistent_full_video_state = latents_video.detach().clone()
        self._persistent_pack_count += 1
        return packed

    @torch.no_grad()
    def update_persistent_video_cache(
        self,
        packed_latents: torch.Tensor,
        step_delta: torch.Tensor,
        scheduler: Any,
        first_frame_latents: torch.Tensor,
    ) -> None:
        """Advance background with Step0 velocity and scatter current sparse state.

        This updates latent state only. Unselected future tokens do not execute
        Transformer queries/FFNs or become additional sparse attention keys.
        """
        if not self.velocity_cache:
            return
        if (self.current_step is None or self.is_dense_step()
                or self._persistent_full_video_state is None
                or self._step0_video_velocity is None
                or self.video_grid is None or self._fixed_video_keep_positions is None):
            raise RuntimeError("Persistent background update requires the Step0 state and velocity cache")
        if self._background_cache_update_steps != [s for s in range(self.current_step) if s not in self.dense_steps]:
            raise RuntimeError("Persistent background cache must be advanced once per sparse step")
        full = scheduler.step(self._step0_video_velocity, step_delta, self._persistent_full_video_state)
        gt, gh, gw = self.video_grid
        pt, ph, pw = (int(v) for v in self.mot.mixtures["video"].patch_size)
        b, c = full.shape[:2]
        patches = (full.reshape(b, c, gt, pt, gh, ph, gw, pw)
                   .permute(0, 2, 4, 6, 3, 5, 7, 1)
                   .reshape(b, gt * gh * gw, pt, ph, pw, c).clone())
        patches.index_copy_(1, self._fixed_video_keep_positions, packed_latents)
        full = (patches.reshape(b, gt, gh, gw, pt, ph, pw, c)
                .permute(0, 7, 1, 4, 2, 5, 3, 6).reshape_as(full))
        full[:, :, :1] = first_frame_latents
        self._persistent_full_video_state = full
        self._background_cache_update_steps.append(self.current_step)
        self._velocity_replaced_elements += b * c * (2 * gh * gw - packed_latents.shape[1]) * pt * ph * pw

    @torch.no_grad()
    def prepare_persistent_mot_inputs(
        self,
        *,
        packed_video_state: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, Mapping[str, torch.Tensor] | None],
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """Build Step1-N inputs without reconstructing the full video latent grid."""
        if self.current_step is None or self.is_dense_step():
            raise RuntimeError("Persistent sparse inputs require a Step1-N denoise step")
        if (
            self._fixed_sparse_attention_mask is None
            or self._persistent_freqs_all is None
            or self._persistent_context_all is None
        ):
            raise RuntimeError("Version1 persistent sparse workspace is unavailable")
        video_expert = self.mot.mixtures["video"]
        action_expert = self.mot.mixtures["action"]
        patch_t, patch_h, patch_w = (int(value) for value in video_expert.patch_size)
        if tuple(packed_video_state.shape[2:5]) != (patch_t, patch_h, patch_w):
            raise ValueError(f"Unexpected packed video patch shape {tuple(packed_video_state.shape)}")
        if packed_video_state.shape[1] == 0:
            video_tokens = packed_video_state.new_empty((1, 0, video_expert.hidden_dim))
            video_t = video_tokens
            video_t_mod = packed_video_state.new_empty((1, 0, 6, video_expert.hidden_dim))
        else:
            video_tokens, video_t, video_t_mod = self._embed_packed_video(packed_video_state, timestep_video)

        action_tokens = action_expert.action_encoder(latents_action)
        action_t = action_expert.time_embedding(
            sinusoidal_embedding_1d(action_expert.freq_dim, timestep_action)
        )
        action_t_mod = action_expert.time_projection(action_t).unflatten(1, (6, action_expert.hidden_dim))
        return (
            {"video": video_tokens, "action": action_tokens},
            self._fixed_sparse_attention_mask,
            dict(self._persistent_freqs_all), dict(self._persistent_context_all),
            {"video": video_t_mod, "action": action_t_mod}, video_t,
        )

    def _embed_packed_video(self, packed_video_state, timestep_video):
        video_expert = self.mot.mixtures["video"]
        patch_input = packed_video_state.permute(0, 1, 5, 2, 3, 4).reshape(
            int(packed_video_state.shape[0]), int(packed_video_state.shape[1]), -1
        )
        patch_weight = video_expert.patch_embedding.weight.reshape(
            int(video_expert.patch_embedding.weight.shape[0]), -1
        )
        video_tokens = F.linear(
            patch_input,
            patch_weight,
            video_expert.patch_embedding.bias,
        )
        video_t = video_expert.time_embedding(
            sinusoidal_embedding_1d(video_expert.freq_dim, timestep_video)
        ).unsqueeze(1).expand(-1, int(video_tokens.shape[1]), -1)
        video_t_mod = video_expert.time_projection(video_t).unflatten(
            2, (6, video_expert.hidden_dim)
        )

        return video_tokens, video_t, video_t_mod

    @torch.no_grad()
    def post_persistent_video(self, tokens: torch.Tensor, video_t: torch.Tensor) -> torch.Tensor:
        video_expert = self.mot.mixtures["video"]
        patch_t, patch_h, patch_w = (int(value) for value in video_expert.patch_size)
        if tokens.shape[1] == 0:
            channels = video_expert.head.head.out_features // (patch_t * patch_h * patch_w)
            return tokens.new_empty((1, 0, patch_t, patch_h, patch_w, channels))
        prediction = video_expert.head(tokens, video_t)
        return prediction.reshape(
            int(prediction.shape[0]),
            int(prediction.shape[1]),
            patch_t,
            patch_h,
            patch_w,
            -1,
        )

    def mark_persistent_step_complete(self) -> None:
        if self.current_step is None or self.is_dense_step():
            raise RuntimeError("No active persistent sparse denoise step")
        self._completed_steps.append(self.current_step)

    @torch.no_grad()
    def complete_mot_forward(self, tokens_all: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.current_step is None:
            raise RuntimeError("Version1 has no active denoise step")
        result = dict(tokens_all)
        if self.is_dense_step():
            if sorted(self._step0_profiles) != list(range(30)):
                raise RuntimeError(f"Incomplete global B0-B29 profile: {sorted(self._step0_profiles)}")
            raw = torch.stack([self._step0_profiles[block] for block in range(30)])
            self.plan = build_fastwam_joint_version1_plan(
                raw,
                top_block_count=self.top_block_count,
                core_token_count=self.core_token_count,
                stable_token_count=self.stable_token_count,
                stable_cv_penalty=self.stable_cv_penalty,
                balanced_budget=self.balanced_budget,
            )
            if self.selection_strategy == "random":
                if self.stable_token_count != 0:
                    raise ValueError("Random selection uses a zero Stable budget")
                self.plan = apply_random_fastwam_joint_version1_selection(
                    self.plan,
                    token_count=self.core_token_count,
                    seed=self.random_selection_seed,
                )
            self._prepare_sparse_workspace()
        else:
            if self._full_video_side_buffer is None or self._video_keep_positions is None:
                raise RuntimeError("Version1 sparse video restoration state is missing")
            sparse_video = result["video"]
            restored = self._full_video_side_buffer.clone()
            restored.index_copy_(1, self._video_keep_positions, sparse_video)
            result["video"] = restored
            self._full_video_side_buffer = None
            self._video_keep_positions = None
        return result

    @torch.no_grad()
    def process_video_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.current_step is None or self.video_grid is None:
            raise RuntimeError("Version1 video prediction has no active denoise step")
        if self.is_dense_step():
            if self.velocity_cache or not self.persistent_action_only:
                self._step0_video_velocity = prediction.detach().clone()
                self._background_cache_source_step = self.current_step
            result = prediction
        elif self.velocity_cache:
            if self.plan is None or self._step0_video_velocity is None:
                raise RuntimeError("Version1 Step-0 video velocity cache is unavailable")
            grid_t, grid_h, grid_w = self.video_grid
            if prediction.ndim != 5:
                raise ValueError(f"Expected video prediction [B,C,T,H,W], got {tuple(prediction.shape)}")
            if any(
                int(prediction.shape[-3 + axis]) % size
                for axis, size in enumerate((grid_t, grid_h, grid_w))
            ):
                raise ValueError("Video prediction shape is not divisible by the token grid")
            # Condition tokens do not execute the Step1-N sparse body either.
            # Reuse their valid Step0 velocity just like unselected future tokens.
            token_roi = torch.zeros((grid_t, grid_h, grid_w), dtype=torch.bool, device=prediction.device)
            token_roi[1:] = self.plan.fixed_masks.to(device=prediction.device).reshape(2, grid_h, grid_w)
            roi = token_roi
            for dim, repeat in enumerate(
                (
                    int(prediction.shape[-3]) // grid_t,
                    int(prediction.shape[-2]) // grid_h,
                    int(prediction.shape[-1]) // grid_w,
                )
            ):
                roi = roi.repeat_interleave(repeat, dim=dim)
            view = roi.unsqueeze(0).unsqueeze(0)
            result = torch.where(view, prediction, self._step0_video_velocity.to(prediction))
            pixels_per_token = math.prod(
                (
                    int(prediction.shape[-3]) // grid_t,
                    int(prediction.shape[-2]) // grid_h,
                    int(prediction.shape[-1]) // grid_w,
                )
            )
            selected_elements = (
                int(prediction.shape[0])
                * int(prediction.shape[1])
                * 2
                * (self.core_token_count + self.stable_token_count)
                * pixels_per_token
            )
            self._velocity_replaced_elements += int(prediction.numel()) - selected_elements
        else:
            result = prediction

        self._completed_steps.append(self.current_step)
        return result

    def summary(self) -> dict[str, Any]:
        if self.plan is None:
            raise RuntimeError("Version1 summary requested before plan finalization")
        selected = [int(value) for value in self.plan.selected_blocks_ranked.detach().cpu().tolist()]
        dense_tokens = 326
        sparse_video_query_tokens = 2 * (self.core_token_count + self.stable_token_count)
        sparse_query_tokens = sparse_video_query_tokens + 32
        sparse_key_value_tokens = 98 + sparse_query_tokens
        graph_manager = getattr(self.mot, "_version1_sparse_graph_manager", None)
        return {
            "schema_version": 3,
            "strategy_family": STRATEGY_FAMILY,
            "strategy_version": (
                "fastwam-joint-version1-global-b0-b29-"
                f"core{self.core_token_count}-stable{self.stable_token_count}-"
                f"k{self.core_token_count + self.stable_token_count}"
                f"{'-dense-velocity-fill' if self.persistent_action_only and self.velocity_cache else ''}-{ACTION_WEIGHTING}"
            ),
            "profile_step": 0,
            "profile_blocks": list(range(30)),
            "top_block_selection_scope": "global_B0_B29",
            "selected_blocks_ranked": selected,
            "action_alignment": {"L1": [0, 16], "L2": [16, 32]},
            "action_weighting": ACTION_WEIGHTING,
            "action_weights_per_future": list(ACTION_WEIGHTS),
            "action_weighting_scope": "Step0 profile aggregation only; not action outputs or attention logits",
            "selection_score": "action-weighted raw complete-softmax attention probability",
            "visualization_score": "per-future-latent spatial normalization",
            "core_tokens_per_future": self.core_token_count,
            "selection_policy": (
                "random_uniform_without_replacement" if self.selection_strategy == "random" else
                "stable_first_balanced_experiment" if self.balanced_budget else "core_first_shared_stable"
            ),
            "selection_strategy": self.selection_strategy,
            "random_selection_seed": self.random_selection_seed if self.selection_strategy == "random" else None,
            "stable_tokens_shared": self.stable_token_count,
            "fixed_tokens_per_future": self.core_token_count + self.stable_token_count,
            "dense_total_tokens": dense_tokens,
            "sparse_video_tokens": sparse_key_value_tokens - 32,
            "sparse_total_tokens": sparse_key_value_tokens,
            "sparse_total_retained_ratio": sparse_key_value_tokens / dense_tokens,
            "condition_kv_cache_tokens": 98,
            "sparse_video_query_tokens": sparse_video_query_tokens,
            "sparse_query_tokens": sparse_query_tokens,
            "sparse_key_value_tokens": sparse_key_value_tokens,
            "sparse_query_retained_ratio": sparse_query_tokens / dense_tokens,
            "sparse_key_value_retained_ratio": sparse_key_value_tokens / dense_tokens,
            "sparse_steps": [s for s in range(self.num_steps) if s not in self.dense_steps],
            "dense_steps": list(self.dense_steps),
            "dense_steps_executed": list(self._dense_steps_seen),
            "graph_invariant_refresh_steps": list(self._graph_invariant_refresh_steps),
            "profile_steps": list(self.dense_steps),
            "completed_steps": list(self._completed_steps),
            "kernel_storage_validation_steps": list(self._kernel_storage_validation_steps),
            "full_latent_cache": self._persistent_full_video_state is not None,
            "background_cache_source_step": self._background_cache_source_step if self._persistent_full_video_state is not None else None,
            "background_cache_update_steps": list(self._background_cache_update_steps),
            "background_cache_policy": ("step0_dense_velocity" if len(self.dense_steps) == 1 else "latest_dense_refresh_velocity") if self._persistent_full_video_state is not None else None,
            "background_cache_scope": "latent_state_only; unselected tokens excluded from sparse attention",
            "sparse_blocks": list(range(30)),
            "single_pack_per_sparse_step": False,
            "persistent_action_only": self.persistent_action_only,
            "single_pack_per_request": self._persistent_pack_count == 1,
            "persistent_pack_count": self._persistent_pack_count,
            "single_pack_per_dense_refresh": self._persistent_pack_count == len(self.dense_steps),
            "restore_full_video_tokens": not self.persistent_action_only,
            "cache_unselected_video_velocity": self.velocity_cache,
            "cross_attention_kv_cached": self._cross_attention_kv_by_block is not None,
            "cross_attention_kv_reused_from_step0": self._cross_attention_kv_reused_from_step0,
            "fused_step0_profile_enabled": self._fused_step0_profile_enabled,
            "fused_step0_profile_used": self._fused_step0_profile_used,
            "fused_step0_profile_error": self._fused_step0_profile_error,
            "torch_compile_requested": self.torch_compile,
            "torch_compile_boundary": (
                "step1_N_complete_B0_B29_sparse_stack" if self.torch_compile else None
            ),
            "torch_compile_fallback_error": self.torch_compile_fallback_error,
            "cuda_graph_requested": self.cuda_graph,
            "cuda_graph_fallback_error": self.graph_fallback_error,
            "cuda_graph_capture_count": int(getattr(graph_manager, "capture_count", 0)),
            "cuda_graph_replay_count": int(getattr(graph_manager, "replay_count", 0)),
            "cuda_graph_capture_validation": getattr(graph_manager, "capture_validation", None),
            "velocity_cache": self.velocity_cache,
            "velocity_replaced_elements": self._velocity_replaced_elements,
        }

    def save_artifacts(self, output_dir: Path, *, metadata: Mapping[str, Any]) -> None:
        if self.plan is None:
            raise RuntimeError("Cannot save Version1 artifacts before plan finalization")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        plan = self.plan
        summary = {**self.summary(), **{str(key): value for key, value in metadata.items()}}
        (output_dir / "version1_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "version1_masks_and_profiles.npz",
            raw_profiles=plan.raw_profiles.detach().cpu().numpy(),
            spatial_profiles=plan.spatial_profiles.detach().cpu().numpy(),
            block_mass=plan.block_mass.detach().cpu().numpy(),
            block_entropy=plan.block_entropy.detach().cpu().numpy(),
            block_quality=plan.block_quality.detach().cpu().numpy(),
            selected_blocks_ranked=plan.selected_blocks_ranked.detach().cpu().numpy(),
            selected_block_weights=plan.selected_block_weights.detach().cpu().numpy(),
            core_scores=plan.core_scores.detach().cpu().numpy(),
            core_masks=plan.core_masks.detach().cpu().numpy(),
            global_latent_scores=plan.global_latent_scores.detach().cpu().numpy(),
            stable_score=plan.stable_score.detach().cpu().numpy(),
            stable_mask=plan.stable_mask.detach().cpu().numpy(),
            fixed_masks=plan.fixed_masks.detach().cpu().numpy(),
            attention_mass_retention=plan.attention_mass_retention.detach().cpu().numpy(),
        )
        with (output_dir / "block_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "block",
                "selected_rank",
                "block_mass",
                "block_entropy",
                "block_quality",
                "L1_fixed_attention_mass_retention",
                "L2_fixed_attention_mass_retention",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            ranking = {
                int(block): rank
                for rank, block in enumerate(plan.selected_blocks_ranked.detach().cpu().tolist(), start=1)
            }
            for block in range(30):
                writer.writerow(
                    {
                        "block": block,
                        "selected_rank": ranking.get(block, ""),
                        "block_mass": float(plan.block_mass[block]),
                        "block_entropy": float(plan.block_entropy[block]),
                        "block_quality": float(plan.block_quality[block]),
                        "L1_fixed_attention_mass_retention": float(plan.attention_mass_retention[block, 0]),
                        "L2_fixed_attention_mass_retention": float(plan.attention_mass_retention[block, 1]),
                    }
                )
