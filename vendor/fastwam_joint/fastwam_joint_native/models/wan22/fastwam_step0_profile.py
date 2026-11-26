from __future__ import annotations

import math

import torch

# Per-latent local action groups: 0:4, 4:8, 8:12, 12:16.
ACTION_WEIGHTING = "aligned16-group4-middle2x"
ACTION_WEIGHT_NUMERATORS = (1,) * 4 + (2,) * 8 + (1,) * 4
ACTION_WEIGHT_DENOMINATOR = sum(ACTION_WEIGHT_NUMERATORS)
ACTION_WEIGHTS = tuple(w / ACTION_WEIGHT_DENOMINATOR for w in ACTION_WEIGHT_NUMERATORS)


def weighted_aligned_action_future_profile(head_mean: torch.Tensor) -> torch.Tensor:
    """Weight already-softmaxed, head-averaged [32,2,98] attention.

    Preserve complete-softmax mass; spatial normalization belongs downstream.
    L1 uses actions 0:16, L2 uses 16:32, with identical local weights.
    """
    if tuple(head_mean.shape) != (32, 2, 98):
        raise ValueError(f"Expected head-mean attention [32,2,98], got {tuple(head_mean.shape)}")
    weights = head_mean.new_tensor(ACTION_WEIGHT_NUMERATORS).view(16, 1)
    aligned = torch.stack((head_mean[:16, 0], head_mean[16:, 1]))
    return (aligned * weights).sum(dim=1) / ACTION_WEIGHT_DENOMINATOR

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in non-CUDA deployments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _aligned_profile_partial_kernel(
        q_ptr,
        k_ptr,
        lse_ptr,
        partial_ptr,
        q_stride_s: tl.constexpr,
        k_stride_s: tl.constexpr,
        lse_stride_h: tl.constexpr,
        lse_stride_s: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEADS_PER_GROUP: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        latent = tl.program_id(0)
        spatial_block = tl.program_id(1)
        head_group = tl.program_id(2)
        spatial_local = tl.arange(0, BLOCK_N)
        spatial = spatial_block * BLOCK_N + spatial_local
        valid_spatial = spatial < 98
        actions = tl.arange(0, 16)
        action_weights = tl.where((actions >= 4) & (actions < 12), 2.0, 1.0)
        dims = tl.arange(0, HEAD_DIM)
        accumulated = tl.zeros((BLOCK_N,), tl.float32)
        for head_offset in range(HEADS_PER_GROUP):
            head = head_group * HEADS_PER_GROUP + head_offset
            q_token = 294 + latent * 16 + actions
            q_offsets = q_token[:, None] * q_stride_s + head * HEAD_DIM + dims[None, :]
            key_token = 98 + latent * 98 + spatial
            k_offsets = (
                key_token[None, :] * k_stride_s + head * HEAD_DIM + dims[:, None]
            )
            q = tl.load(q_ptr + q_offsets)
            k = tl.load(k_ptr + k_offsets, mask=valid_spatial[None, :], other=0.0)
            logits = tl.dot(q, k) * scale
            lse_offsets = head * lse_stride_h + q_token * lse_stride_s
            logsumexp = tl.load(lse_ptr + lse_offsets)
            accumulated += tl.sum(
                tl.exp(logits - logsumexp[:, None]) * action_weights[:, None], axis=0
            )
        partial_offset = (
            (latent * 7 + spatial_block) * 6 + head_group
        ) * BLOCK_N + spatial_local
        tl.store(partial_ptr + partial_offset, accumulated, mask=valid_spatial)

    @triton.jit
    def _aligned_profile_reduce_kernel(
        partial_ptr,
        output_ptr,
        ACTION_WEIGHT_SUM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        latent = tl.program_id(0)
        spatial_block = tl.program_id(1)
        spatial_local = tl.arange(0, BLOCK_N)
        spatial = spatial_block * BLOCK_N + spatial_local
        valid = spatial < 98
        total = tl.zeros((BLOCK_N,), tl.float32)
        for group in range(6):
            offset = (
                (latent * 7 + spatial_block) * 6 + group
            ) * BLOCK_N + spatial_local
            total += tl.load(partial_ptr + offset, mask=valid, other=0.0)
        tl.store(
            output_ptr + latent * 98 + spatial,
            total / (24.0 * ACTION_WEIGHT_SUM),
            mask=valid,
        )


def aligned_action_future_profile_from_lse(
    *,
    q_cat: torch.Tensor,
    k_cat: torch.Tensor,
    attention_logsumexp: torch.Tensor,
    partial_workspace: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Reduce action-weighted aligned future mass using the main attention LSE."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not q_cat.is_cuda or not k_cat.is_cuda or not attention_logsumexp.is_cuda:
        raise ValueError("Fused Version1 profile requires CUDA tensors")
    if tuple(q_cat.shape) != (1, 326, 3072) or tuple(k_cat.shape) != (1, 326, 3072):
        raise ValueError(
            "Fused Version1 profile expects Q/K [1,326,3072], "
            f"got {tuple(q_cat.shape)}/{tuple(k_cat.shape)}"
        )
    if (
        tuple(attention_logsumexp.shape[:2]) != (1, 24)
        or int(attention_logsumexp.shape[2]) < 326
    ):
        raise ValueError(
            "Fused Version1 profile expects LSE [1,24,>=326], "
            f"got {tuple(attention_logsumexp.shape)}"
        )
    if tuple(partial_workspace.shape) != (2, 7, 6, 16):
        raise ValueError(
            f"Unexpected profile workspace {tuple(partial_workspace.shape)}"
        )
    if tuple(output.shape) != (2, 98) or output.dtype != torch.float32:
        raise ValueError(
            f"Unexpected profile output {tuple(output.shape)}/{output.dtype}"
        )

    _aligned_profile_partial_kernel[(2, 7, 6)](
        q_cat,
        k_cat,
        attention_logsumexp,
        partial_workspace,
        q_cat.stride(1),
        k_cat.stride(1),
        attention_logsumexp.stride(1),
        attention_logsumexp.stride(2),
        1.0 / math.sqrt(128),
        BLOCK_N=16,
        HEADS_PER_GROUP=4,
        HEAD_DIM=128,
        num_warps=4,
    )
    _aligned_profile_reduce_kernel[(2, 7)](
        partial_workspace,
        output,
        ACTION_WEIGHT_SUM=ACTION_WEIGHT_DENOMINATOR,
        BLOCK_N=16,
        num_warps=4,
    )
    return output


def efficient_attention_with_lse(
    *,
    q_cat: torch.Tensor,
    k_cat: torch.Tensor,
    v_cat: torch.Tensor,
    attention_mask: torch.Tensor,
    num_heads: int,
    controller,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PyTorch's current efficient-attention backend and expose its LSE."""
    if tuple(q_cat.shape[:2]) != (1, 326) or int(num_heads) != 24:
        raise ValueError(
            f"Unexpected Step0 attention shape {tuple(q_cat.shape)}/{num_heads}"
        )
    head_dim = int(q_cat.shape[-1]) // int(num_heads)
    q = q_cat.view(1, 326, int(num_heads), head_dim).permute(0, 2, 1, 3)
    k = k_cat.view(1, 326, int(num_heads), head_dim).permute(0, 2, 1, 3)
    v = v_cat.view(1, 326, int(num_heads), head_dim).permute(0, 2, 1, 3)
    bias = getattr(controller, "_step0_attention_bias", None)
    if bias is None or bias.device != q.device or bias.dtype != q.dtype:
        storage = torch.zeros(
            (1, int(num_heads), 326, 328), dtype=q.dtype, device=q.device
        )
        bias = storage[..., :326]
        bias.masked_fill_(
            ~attention_mask.to(device=q.device, dtype=torch.bool).view(1, 1, 326, 326),
            float("-inf"),
        )
        controller._step0_attention_bias_storage = storage
        controller._step0_attention_bias = bias
    output, logsumexp, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q,
        k,
        v,
        bias,
        True,
        0.0,
        False,
    )
    output = output.permute(0, 2, 1, 3).reshape_as(q_cat)
    return output, logsumexp
