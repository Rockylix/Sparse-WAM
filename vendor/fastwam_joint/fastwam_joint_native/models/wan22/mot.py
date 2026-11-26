from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastwam_step0_profile import efficient_attention_with_lse
from .wan_video_dit import flash_attention, modulate, rope_apply
from logging import getLogger as get_logger

logger = get_logger(__name__)


def _version1_sparse_ffn_impl(
    x: torch.Tensor,
    gate_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mlp_input = F.layer_norm(x, (x.shape[-1],), eps=eps)
    mlp_input = mlp_input * (1.0 + scale_mlp) + shift_mlp
    hidden = F.linear(mlp_input, weight1, bias1)
    hidden = F.gelu(hidden, approximate="tanh")
    output = F.linear(hidden, weight2, bias2)
    return x + gate_mlp * output


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @contextmanager
    def _nvtx_profile_range(self, name: str) -> Iterator[None]:
        """Emit detailed MoT ranges only for an explicitly profiled request."""
        label = getattr(self, "nvtx_profile_label", None)
        if not label:
            yield
            return
        if getattr(self, "nvtx_profile_detail", "full") == "coarse" and (
            name.startswith("B") or name in ("self_attention_output", "cross_attention", "ffn")
        ):
            yield
            return
        step = getattr(self, "nvtx_profile_step", None)
        step_label = "stepNA" if step is None else f"step{int(step):02d}"
        torch.cuda.nvtx.range_push(f"{label}/{step_label}/{name}")
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    def _apply_expert_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
        version1_layer_index: Optional[int] = None,
        version1_expert_name: Optional[str] = None,
    ) -> torch.Tensor:
        with self._nvtx_profile_range("self_attention_output"):
            x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        with self._nvtx_profile_range("cross_attention"):
            if context_payload is not None:
                context = context_payload.get("context")
                if context is not None:
                    context_mask = context_payload.get("mask")
                    if context_mask is not None and context_mask.dim() == 3:
                        context_mask = context_mask.unsqueeze(1)
                    controller = getattr(self, "version1_sparse_controller", None)
                    capture_step0_kv = bool(
                        controller is not None
                        and getattr(controller, "current_step", None) in getattr(controller, "dense_steps", (0,))
                        and version1_layer_index is not None
                        and version1_expert_name is not None
                    )
                    if capture_step0_kv:
                        # Retain the exact context K/V consumed by dense Step0 so
                        # sparse-workspace finalization does not project it again.
                        cross_attn = block.cross_attn
                        normalized_x = block.norm3(x)
                        cross_q = cross_attn.norm_q(cross_attn.q(normalized_x))
                        cross_k = cross_attn.norm_k(cross_attn.k(context))
                        cross_v = cross_attn.v(context)
                        controller.capture_step0_cross_attention_kv(
                            layer_index=int(version1_layer_index),
                            expert_name=str(version1_expert_name),
                            k=cross_k,
                            v=cross_v,
                        )
                        cross_output = flash_attention(
                            q=cross_q,
                            k=cross_k,
                            v=cross_v,
                            num_heads=cross_attn.num_heads,
                            ctx_mask=context_mask,
                        )
                        x = x + cross_attn.o(cross_output)
                    else:
                        x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        with self._nvtx_profile_range("ffn"):
            mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
            x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _apply_version1_sparse_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
        cached_cross_attention_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        with self._nvtx_profile_range("self_attention_output"):
            x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
        with self._nvtx_profile_range("cross_attention"):
            if context_payload is not None:
                context = context_payload.get("context")
                if context is not None:
                    context_mask = context_payload.get("mask")
                    if context_mask is not None and context_mask.dim() == 3:
                        context_mask = context_mask.unsqueeze(1)
                    if cached_cross_attention_kv is None:
                        cross_output = block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
                    else:
                        cross_k, cross_v = cached_cross_attention_kv
                        cross_attn = block.cross_attn
                        cross_q = cross_attn.norm_q(cross_attn.q(block.norm3(x)))
                        cross_output = flash_attention(
                            q=cross_q,
                            k=cross_k,
                            v=cross_v,
                            num_heads=cross_attn.num_heads,
                            ctx_mask=context_mask,
                        )
                        cross_output = cross_attn.o(cross_output)
                    x = x + cross_output

        with self._nvtx_profile_range("ffn"):
            if (
                isinstance(block.ffn, nn.Sequential)
                and len(block.ffn) == 3
                and isinstance(block.ffn[0], nn.Linear)
                and isinstance(block.ffn[2], nn.Linear)
                and block.ffn[0].bias is not None
                and block.ffn[2].bias is not None
            ):
                return _version1_sparse_ffn_impl(
                    x,
                    gate_mlp,
                    shift_mlp,
                    scale_mlp,
                    block.ffn[0].weight,
                    block.ffn[0].bias,
                    block.ffn[2].weight,
                    block.ffn[2].bias,
                    float(block.norm2.eps),
                )
            mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
            return block.gate(x, gate_mlp, block.ffn(mlp_input))

    @staticmethod
    @torch.no_grad()
    def _prepare_version1_fused_qkv(self_attn: nn.Module, *, allow_rebuild: bool = False) -> bool:
        """Prepare one inference-only BF16 QKV weight without persistent duplication."""
        if self_attn.training:
            raise RuntimeError("Version1 fused QKV is an irreversible inference-only optimization")
        if hasattr(self_attn, "_version1_fused_qkv_weight"):
            fused_weight = self_attn._version1_fused_qkv_weight
            fused_bias = self_attn._version1_fused_qkv_bias
            projections = (self_attn.q, self_attn.k, self_attn.v)
            width = int(projections[0].out_features)
            invalid = []
            for index, projection in enumerate(projections):
                expected_weight_offset = fused_weight.storage_offset() + index * width * int(
                    projection.in_features
                )
                if not (
                    projection.weight.shape == (width, int(projection.in_features))
                    and projection.weight.dtype == fused_weight.dtype
                    and projection.weight.device == fused_weight.device
                    and projection.weight.untyped_storage().data_ptr()
                    == fused_weight.untyped_storage().data_ptr()
                    and projection.weight.storage_offset() == expected_weight_offset
                ):
                    invalid.append(f"weight[{index}]")
                if fused_bias is not None:
                    expected_bias_offset = fused_bias.storage_offset() + index * width
                    if not (
                        projection.bias is not None
                        and projection.bias.dtype == fused_bias.dtype
                        and projection.bias.device == fused_bias.device
                        and projection.bias.untyped_storage().data_ptr()
                        == fused_bias.untyped_storage().data_ptr()
                        and projection.bias.storage_offset() == expected_bias_offset
                    ):
                        invalid.append(f"bias[{index}]")
            if invalid:
                if not allow_rebuild:
                    raise RuntimeError(
                        "Version1 fused QKV storage was invalidated by model.to(), assign=True load, "
                        "or parameter rebinding; rebuild the inference model before reuse; invalid="
                        + ",".join(invalid)
                    )
                del self_attn._version1_fused_qkv_weight
                del self_attn._version1_fused_qkv_bias
            else:
                return True
        projections = (self_attn.q, self_attn.k, self_attn.v)
        if not all(isinstance(projection, nn.Linear) for projection in projections):
            return False
        output_dims = [int(projection.out_features) for projection in projections]
        if len(set(output_dims)) != 1:
            return False
        has_bias = [projection.bias is not None for projection in projections]
        if len(set(has_bias)) != 1:
            return False

        fused_weight = torch.cat([projection.weight.detach() for projection in projections], dim=0)
        fused_bias = (
            torch.cat([projection.bias.detach() for projection in projections], dim=0)
            if has_bias[0]
            else None
        )
        width = output_dims[0]
        for index, projection in enumerate(projections):
            projection.weight = nn.Parameter(
                fused_weight.narrow(0, index * width, width),
                requires_grad=False,
            )
            if fused_bias is not None:
                projection.bias = nn.Parameter(
                    fused_bias.narrow(0, index * width, width),
                    requires_grad=False,
                )
        self_attn._version1_fused_qkv_weight = fused_weight
        self_attn._version1_fused_qkv_bias = fused_bias
        return True

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        *,
        version1_fused_qkv: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        # Version1 BF16 QKV: use one wide GEMM so small-M sparse inputs expose
        # enough N tiles to occupy the GPU and avoid two kernel launches.
        sa = block.self_attn
        used_version1_fused_qkv = bool(version1_fused_qkv)
        if used_version1_fused_qkv:
            if not hasattr(sa, "_version1_fused_qkv_weight"):
                raise RuntimeError("Version1 fused QKV was not prepared before sparse execution")
            qkv = F.linear(
                attn_input,
                sa._version1_fused_qkv_weight,
                sa._version1_fused_qkv_bias,
            )
            q_raw, k_raw, v = qkv.chunk(3, dim=-1)
            q = sa.norm_q(q_raw)
            k = sa.norm_k(k_raw)
        else:
            q = sa.norm_q(sa.q(attn_input))
            k = sa.norm_k(sa.k(attn_input))
            v = sa.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
        version1_layer_index: Optional[int] = None,
        version1_expert_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
            _version1_layer_index=version1_layer_index,
            _version1_expert_name=version1_expert_name,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
                version1_layer_index=_version1_layer_index,
                version1_expert_name=_version1_expert_name,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        kv_cache, _ = self.prefill_video_cache_with_tokens(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
        )
        return kv_cache

    def prefill_video_cache_with_tokens(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Prefill video branch and return both per-layer K/V and final tokens."""
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache, x

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def prepare_version1_sparse_kernels(self, *, allow_rebuild: bool = False) -> None:
        """Materialize inference-only wide-QKV weights before graph capture."""
        storage_changed = False
        for expert in self.mixtures.values():
            for block in expert.blocks:
                previous = getattr(block.self_attn, "_version1_fused_qkv_weight", None)
                previous_ptr = (
                    previous.untyped_storage().data_ptr()
                    if isinstance(previous, torch.Tensor)
                    else None
                )
                self._prepare_version1_fused_qkv(block.self_attn, allow_rebuild=allow_rebuild)
                current = getattr(block.self_attn, "_version1_fused_qkv_weight", None)
                current_ptr = (
                    current.untyped_storage().data_ptr()
                    if isinstance(current, torch.Tensor)
                    else None
                )
                storage_changed = storage_changed or previous_ptr != current_ptr
        if storage_changed:
            self._version1_sparse_kernel_generation = (
                int(getattr(self, "_version1_sparse_kernel_generation", 0)) + 1
            )
            manager = getattr(self, "_version1_sparse_graph_manager", None)
            if manager is not None:
                manager.reset()

    def version1_sparse_body(self, *, torch_compile: bool):
        """Return the eager or whole-stack compiled fixed-shape sparse body.

        Step0 remains eager because it profiles attention and builds request
        caches.  For Step1-N, the compile boundary spans the complete B0-B29
        sparse Transformer stack, matching the Edge Version1 methodology.
        """
        if not torch_compile:
            # Bound-method lookup creates a new Python object each time. Keep
            # its identity stable so the graph manager does not recapture at
            # every denoise step/request when running without torch.compile.
            eager = getattr(self, "_version1_eager_sparse_body", None)
            if eager is None:
                eager = self._forward_version1_sparse_body
                self._version1_eager_sparse_body = eager
            return eager
        generation = int(getattr(self, "_version1_sparse_kernel_generation", 0))
        compiled = getattr(self, "_version1_compiled_sparse_body", None)
        compiled_generation = getattr(self, "_version1_compiled_sparse_body_generation", None)
        if compiled is None or compiled_generation != generation:
            compiled = torch.compile(
                self._forward_version1_sparse_body,
                fullgraph=True,
                dynamic=False,
                mode=None,
            )
            self._version1_compiled_sparse_body = compiled
            self._version1_compiled_sparse_body_generation = generation
        return compiled

    def _forward_version1_sparse_body(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        condition_kv_by_block: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        cross_attention_kv_by_block: Optional[
            tuple[Dict[str, tuple[torch.Tensor, torch.Tensor]], ...]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """Dedicated Step1-N branch for packed future/action queries.

        Query order is ``[selected future, action]``. Key/value order is
        ``[cached condition, selected future, action]``. Only dynamic
        future/action Q/K/V and post blocks are recomputed.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Version1 sparse forward is inference-only")
        if self.expert_order != ["video", "action"]:
            raise ValueError(f"Unexpected Version1 expert order {self.expert_order}")

        tokens_all = {name: embeds_all[name] for name in self.expert_order}
        active_experts = [name for name in self.expert_order if embeds_all[name].shape[1] > 0]
        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in active_experts:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                with self._nvtx_profile_range(f"B{layer_idx:02d}/qkv_{name}"):
                    (
                        q,
                        k,
                        v,
                        residual_x,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        use_gradient_checkpointing,
                    ) = self._build_expert_attention_io(
                        expert=expert,
                        block=block,
                        x=tokens_all[name],
                        freqs=freqs_all[name],
                        t_mod=t_mod_all[name],
                        version1_fused_qkv=True,
                    )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(int(tokens_all[name].shape[1]))
                cached[name] = (
                    block,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                )

            q_cat = torch.cat(q_chunks, dim=1)
            dynamic_k = torch.cat(k_chunks, dim=1)
            dynamic_v = torch.cat(v_chunks, dim=1)
            condition_k, condition_v = condition_kv_by_block[layer_idx]
            k_cat = torch.cat((condition_k, dynamic_k), dim=1)
            v_cat = torch.cat((condition_v, dynamic_v), dim=1)
            if tuple(attention_mask.shape) != (int(q_cat.shape[1]), int(k_cat.shape[1])):
                raise ValueError(
                    f"Version1 rectangular mask {tuple(attention_mask.shape)} does not match "
                    f"Q/K={(int(q_cat.shape[1]), int(k_cat.shape[1]))}"
                )

            with self._nvtx_profile_range(f"B{layer_idx:02d}/attention_rectangular"):
                mixed = self._mixed_attention(
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    attention_mask=attention_mask,
                )

            start = 0
            for name, seq_len in zip(active_experts, seq_lens):
                end = start + seq_len
                (
                    block,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = cached[name]
                with self._nvtx_profile_range(f"B{layer_idx:02d}/post_{name}"):
                    cached_cross_kv = (
                        None
                        if cross_attention_kv_by_block is None
                        else cross_attention_kv_by_block[layer_idx].get(name)
                    )
                    tokens_all[name] = self._apply_version1_sparse_post_block(
                        block=block,
                        residual_x=residual_x,
                        mixed_attn_out=mixed[:, start:end],
                        gate_msa=gate_msa,
                        shift_mlp=shift_mlp,
                        scale_mlp=scale_mlp,
                        gate_mlp=gate_mlp,
                        context_payload=context_all.get(name),
                        cached_cross_attention_kv=cached_cross_kv,
                    )
                start = end

        return tokens_all

    def forward_version1_sparse(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        controller,
    ) -> Dict[str, torch.Tensor]:
        if controller.current_step is None or controller.current_step in getattr(controller, "dense_steps", (0,)):
            raise RuntimeError("Version1 sparse forward requires a post-profile denoise step")
        # Step0 has already validated/rebuilt the shared weights on the final
        # inference device. Weights stay immutable during a request: repeating
        # the 60-block storage walk here leaves a CPU bubble at every step.
        # Keep a lazy preparation path for callers that start at a sparse step.
        if not getattr(controller, "_sparse_kernels_prepared", False):
            self.prepare_version1_sparse_kernels()
            controller._sparse_kernels_prepared = True
            if hasattr(controller, "_kernel_storage_validation_steps"):
                controller._kernel_storage_validation_steps.append(controller.current_step)
        condition_kv = controller.condition_kv_by_block()
        cross_attention_kv = controller.cross_attention_kv_by_block()
        compile_enabled = bool(
            getattr(controller, "torch_compile", False)
            and getattr(controller, "torch_compile_fallback_error", None) is None
        )
        body_fn = self.version1_sparse_body(torch_compile=compile_enabled)
        tokens_all = None
        if (
            controller.cuda_graph
            and controller.graph_fallback_error is None
            and torch.cuda.is_available()
        ):
            from .fastwam_sparse_graph import FastWAMSparseGraphManager

            manager = getattr(self, "_version1_sparse_graph_manager", None)
            if manager is None or manager.body_fn is not body_fn:
                if manager is not None:
                    manager.reset()
                manager = FastWAMSparseGraphManager(
                    self,
                    body_fn=body_fn,
                    backend_signature=("whole_stack_compile" if compile_enabled else "eager"),
                )
                self._version1_sparse_graph_manager = manager
            try:
                capture_count = manager.capture_count
                with self._nvtx_profile_range("graph_capture_or_replay"):
                    tokens_all = manager.run(
                        embeds_all=embeds_all,
                        attention_mask=attention_mask,
                        freqs_all=freqs_all,
                        context_all=context_all,
                        t_mod_all=t_mod_all,
                        condition_kv_by_block=condition_kv,
                        cross_attention_kv_by_block=cross_attention_kv,
                        update_invariants=(controller.current_step - 1) in getattr(controller, "dense_steps", (0,)),
                    )
                if ((controller.current_step - 1) in getattr(controller, "dense_steps", (0,))
                        and hasattr(controller, "_graph_invariant_refresh_steps")):
                    controller._graph_invariant_refresh_steps.append(controller.current_step)
                if manager.capture_count > capture_count:
                    logger.info("Captured FastWAM Version1 sparse CUDA Graph")
            except RuntimeError as exc:
                controller.graph_fallback_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Version1 sparse CUDA Graph failed; using eager sparse body: %s", exc)
                try:
                    torch.cuda.synchronize(device=embeds_all["video"].device)
                except Exception:
                    pass
                manager.reset(clear_diagnostics=False)
        if tokens_all is None:
            body_kwargs = {
                "embeds_all": embeds_all,
                "attention_mask": attention_mask,
                "freqs_all": freqs_all,
                "context_all": context_all,
                "t_mod_all": t_mod_all,
                "condition_kv_by_block": condition_kv,
                "cross_attention_kv_by_block": cross_attention_kv,
            }
            try:
                tokens_all = body_fn(**body_kwargs)
            except RuntimeError as exc:
                if not compile_enabled:
                    raise
                controller.torch_compile_fallback_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Version1 whole-stack compile failed; using eager sparse body: %s", exc)
                tokens_all = self._forward_version1_sparse_body(**body_kwargs)
        if bool(getattr(controller, "persistent_action_only", False)):
            return tokens_all
        with self._nvtx_profile_range("restore"):
            return controller.complete_mot_forward(tokens_all)

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        version1_controller = getattr(self, "version1_sparse_controller", None)
        if version1_controller is not None:
            if getattr(self, "action_future_attention_observer", None) is not None:
                raise RuntimeError("Version1 sparse inference cannot share the read-only attention observer")
            with self._nvtx_profile_range("pack"):
                (
                    embeds_all,
                    attention_mask,
                    freqs_all,
                    context_all,
                    t_mod_all,
                ) = version1_controller.prepare_mot_inputs(
                    embeds_all=embeds_all,
                    attention_mask=attention_mask,
                    freqs_all=freqs_all,
                    context_all=context_all,
                    t_mod_all=t_mod_all,
                    expert_order=self.expert_order,
                )
            if version1_controller.current_step is not None and version1_controller.current_step not in version1_controller.dense_steps:
                return self.forward_version1_sparse(
                    embeds_all=embeds_all,
                    attention_mask=attention_mask,
                    freqs_all=freqs_all,
                    context_all=context_all,
                    t_mod_all=t_mod_all,
                    controller=version1_controller,
                )

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                with self._nvtx_profile_range(f"B{layer_idx:02d}/qkv_{name}"):
                    (
                        q,
                        k,
                        v,
                        residual_x,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                        use_gradient_checkpointing,
                    ) = self._build_expert_attention_io(
                        expert=expert,
                        block=block,
                        x=x,
                        freqs=freqs,
                        t_mod=t_mod,
                    )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            query_seq = q_cat.shape[1]
            key_value_seq = k_cat.shape[1]
            if tuple(attention_mask.shape) != (query_seq, key_value_seq):
                raise ValueError(
                    "Attention mask query/key length mismatch: "
                    f"mask={tuple(attention_mask.shape)} vs Q/K={(query_seq, key_value_seq)}"
                )

            attention_logsumexp = None
            with self._nvtx_profile_range(f"B{layer_idx:02d}/attention"):
                if (
                    version1_controller is not None
                    and version1_controller.current_step in version1_controller.dense_steps
                    and version1_controller._fused_step0_profile_enabled
                ):
                    try:
                        mixed, attention_logsumexp = efficient_attention_with_lse(
                            q_cat=q_cat,
                            k_cat=k_cat,
                            v_cat=v_cat,
                            attention_mask=attention_mask,
                            num_heads=self.num_heads,
                            controller=version1_controller,
                        )
                    except Exception as exc:
                        version1_controller._fused_step0_profile_enabled = False
                        if version1_controller._fused_step0_profile_error is None:
                            version1_controller._fused_step0_profile_error = f"{type(exc).__name__}: {exc}"
                        mixed = self._mixed_attention(
                            q_cat=q_cat,
                            k_cat=k_cat,
                            v_cat=v_cat,
                            attention_mask=attention_mask,
                        )
                else:
                    mixed = self._mixed_attention(
                        q_cat=q_cat,
                        k_cat=k_cat,
                        v_cat=v_cat,
                        attention_mask=attention_mask,
                    )
            if version1_controller is not None:
                with self._nvtx_profile_range(f"B{layer_idx:02d}/profile"):
                    version1_controller.observe_dense_block(
                        layer_index=layer_idx,
                        q_cat=q_cat,
                        k_cat=k_cat,
                        v_cat=v_cat,
                        attention_mask=attention_mask,
                        num_heads=self.num_heads,
                        expert_order=self.expert_order,
                        seq_lens=seq_lens,
                        attention_logsumexp=attention_logsumexp,
                    )
            observer = getattr(self, "action_future_attention_observer", None)
            if observer is not None:
                observer.observe(
                    layer_index=layer_idx,
                    q_cat=q_cat,
                    k_cat=k_cat,
                    v_cat=v_cat,
                    mixed_output=mixed,
                    attention_mask=attention_mask,
                    num_heads=self.num_heads,
                    expert_order=self.expert_order,
                    seq_lens=seq_lens,
                )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                with self._nvtx_profile_range(f"B{layer_idx:02d}/post_{name}"):
                    updated_tokens = self._apply_post_with_optional_checkpoint(
                        block=block,
                        residual_x=cached_expert["residual_x"],
                        gate_msa=cached_expert["gate_msa"],
                        shift_mlp=cached_expert["shift_mlp"],
                        scale_mlp=cached_expert["scale_mlp"],
                        gate_mlp=cached_expert["gate_mlp"],
                        use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                        mixed_slice=mixed_slice,
                        context_payload=context_payload,
                        version1_layer_index=layer_idx,
                        version1_expert_name=name,
                    )

                tokens_all[name] = updated_tokens
                start = end

        if version1_controller is not None:
            with self._nvtx_profile_range("restore_or_finalize"):
                tokens_all = version1_controller.complete_mot_forward(tokens_all)
        return tokens_all
