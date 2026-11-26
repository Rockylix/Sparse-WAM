from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional

import torch


def _tensor_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device.type,
            value.device.index,
        )
    if isinstance(value, dict):
        return tuple((key, _tensor_signature(value[key])) for key in sorted(value))
    if isinstance(value, (tuple, list)):
        return tuple(_tensor_signature(item) for item in value)
    if value is None:
        return None
    raise TypeError(f"Unsupported Version1 graph value {type(value)}")


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if value is None:
        return None
    raise TypeError(f"Unsupported Version1 graph value {type(value)}")


def _copy_tree(destination: Any, source: Any) -> None:
    if isinstance(destination, torch.Tensor):
        destination.copy_(source)
        return
    if isinstance(destination, dict):
        for key in destination:
            _copy_tree(destination[key], source[key])
        return
    if isinstance(destination, (tuple, list)):
        for dst_item, src_item in zip(destination, source):
            _copy_tree(dst_item, src_item)
        return
    if destination is None and source is None:
        return
    raise TypeError(f"Unsupported Version1 graph copy {type(destination)}/{type(source)}")


class FastWAMSparseGraphManager:
    """Model-owned CUDA Graph for the pure fixed-shape Version1 sparse body."""

    def __init__(
        self,
        mot: Any,
        *,
        body_fn: Any | None = None,
        backend_signature: str = "eager",
    ) -> None:
        self.mot = mot
        self.body_fn = body_fn if body_fn is not None else mot._forward_version1_sparse_body
        self.backend_signature = str(backend_signature)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.signature: Any = None
        self.static_embeds: Dict[str, torch.Tensor] | None = None
        self.static_attention_mask: torch.Tensor | None = None
        self.static_freqs: Dict[str, torch.Tensor] | None = None
        self.static_context: Dict[str, Optional[dict]] | None = None
        self.static_t_mod: Dict[str, torch.Tensor] | None = None
        self.static_condition_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None
        self.static_cross_attention_kv: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...] | None = None
        self.static_output: Dict[str, torch.Tensor] | None = None
        self.capture_count = 0
        self.replay_count = 0
        self.capture_validation: dict[str, float] | None = None
        self.last_error: str | None = None
        self.execution_stream_id: int | None = None
        self._run_lock = threading.Lock()

    def reset(self, *, clear_diagnostics: bool = True) -> None:
        if self.graph is not None:
            # All returned outputs are cloned before leaving run(), so no caller
            # retains an alias to graph-owned storage.
            self.graph.reset()
        self.graph = None
        self.signature = None
        self.static_embeds = None
        self.static_attention_mask = None
        self.static_freqs = None
        self.static_context = None
        self.static_t_mod = None
        self.static_condition_kv = None
        self.static_cross_attention_kv = None
        self.static_output = None
        self.execution_stream_id = None
        if clear_diagnostics:
            self.capture_validation = None
            self.last_error = None

    def _make_signature(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        condition_kv_by_block: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        cross_attention_kv_by_block: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...],
    ) -> Any:
        return (
            _tensor_signature(embeds_all),
            _tensor_signature(attention_mask),
            _tensor_signature(freqs_all),
            _tensor_signature(context_all),
            _tensor_signature(t_mod_all),
            _tensor_signature(condition_kv_by_block),
            _tensor_signature(cross_attention_kv_by_block),
            torch.is_autocast_enabled(),
            torch.get_autocast_dtype("cuda") if torch.cuda.is_available() else None,
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
            "fastwam-version1-sparse-body-v2-cross-kv",
            self.backend_signature,
            int(getattr(self.mot, "_version1_sparse_kernel_generation", 0)),
        )

    def _body(self) -> Dict[str, torch.Tensor]:
        assert self.static_embeds is not None
        assert self.static_attention_mask is not None
        assert self.static_freqs is not None
        assert self.static_context is not None
        assert self.static_t_mod is not None
        assert self.static_condition_kv is not None
        assert self.static_cross_attention_kv is not None
        return self.body_fn(
            embeds_all=self.static_embeds,
            attention_mask=self.static_attention_mask,
            freqs_all=self.static_freqs,
            context_all=self.static_context,
            t_mod_all=self.static_t_mod,
            condition_kv_by_block=self.static_condition_kv,
            cross_attention_kv_by_block=self.static_cross_attention_kv,
        )

    def _capture(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        condition_kv_by_block: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        cross_attention_kv_by_block: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...],
        signature: Any,
    ) -> Dict[str, torch.Tensor]:
        device = embeds_all["video"].device
        if device.type != "cuda":
            raise RuntimeError(f"Version1 CUDA Graph requires CUDA tensors, got {device}")
        self.reset()
        with torch.cuda.device(device):
            self.static_embeds = _clone_tree(embeds_all)
            self.static_attention_mask = attention_mask.clone()
            self.static_freqs = _clone_tree(freqs_all)
            self.static_context = _clone_tree(context_all)
            self.static_t_mod = _clone_tree(t_mod_all)
            self.static_condition_kv = _clone_tree(condition_kv_by_block)
            self.static_cross_attention_kv = _clone_tree(cross_attention_kv_by_block)

            current_stream = torch.cuda.current_stream(device=device)
            capture_stream = torch.cuda.Stream(device=device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                warmup_output = self._body()
                warmup_reference = {key: value.clone() for key, value in warmup_output.items()}
            current_stream.wait_stream(capture_stream)
            current_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                static_output = self._body()
            # Capture records work but does not produce a valid first-request
            # output. Replay once before validating or returning Step1.
            graph.replay()
            current_stream.synchronize()
        difference = torch.cat(
            [
                (static_output[key].float() - warmup_reference[key].float()).reshape(-1)
                for key in sorted(static_output)
            ]
        )
        reference = torch.cat([warmup_reference[key].float().reshape(-1) for key in sorted(warmup_reference)])
        validation = {
            "relative_l2": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference).clamp_min(1e-12)
            ),
            "max_absolute_error": float(difference.abs().max()),
        }
        if (
            not all(math.isfinite(value) for value in validation.values())
            or validation["relative_l2"] > 5e-3
            or validation["max_absolute_error"] > 5e-2
        ):
            graph.reset()
            self.capture_validation = validation
            self.last_error = f"CUDA Graph validation failed: {validation}"
            raise RuntimeError(self.last_error)
        self.graph = graph
        self.static_output = static_output
        self.signature = signature
        self.execution_stream_id = int(current_stream.cuda_stream)
        self.capture_count += 1
        self.replay_count += 1
        self.capture_validation = validation
        return _clone_tree(static_output)

    def run(
        self,
        *,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        condition_kv_by_block: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        cross_attention_kv_by_block: tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], ...],
        update_invariants: bool,
    ) -> Dict[str, torch.Tensor]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent FastWAM Version1 CUDA Graph use is unsupported")
        try:
            device = embeds_all["video"].device
            current_stream_id = int(torch.cuda.current_stream(device=device).cuda_stream)
            if self.execution_stream_id is not None and current_stream_id != self.execution_stream_id:
                raise RuntimeError("FastWAM Version1 CUDA Graph must replay on its capture stream")
            signature = self._make_signature(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                condition_kv_by_block=condition_kv_by_block,
                cross_attention_kv_by_block=cross_attention_kv_by_block,
            )
            if self.graph is None or signature != self.signature:
                return self._capture(
                    embeds_all=embeds_all,
                    attention_mask=attention_mask,
                    freqs_all=freqs_all,
                    context_all=context_all,
                    t_mod_all=t_mod_all,
                    condition_kv_by_block=condition_kv_by_block,
                    cross_attention_kv_by_block=cross_attention_kv_by_block,
                    signature=signature,
                )

            assert self.static_embeds is not None
            assert self.static_t_mod is not None
            _copy_tree(self.static_embeds, embeds_all)
            _copy_tree(self.static_t_mod, t_mod_all)
            if update_invariants:
                assert self.static_attention_mask is not None
                assert self.static_freqs is not None
                assert self.static_context is not None
                assert self.static_condition_kv is not None
                assert self.static_cross_attention_kv is not None
                self.static_attention_mask.copy_(attention_mask)
                _copy_tree(self.static_freqs, freqs_all)
                _copy_tree(self.static_context, context_all)
                _copy_tree(self.static_condition_kv, condition_kv_by_block)
                _copy_tree(self.static_cross_attention_kv, cross_attention_kv_by_block)
            assert self.graph is not None and self.static_output is not None
            self.graph.replay()
            self.replay_count += 1
            # A private copy keeps graph-owned outputs safe from a subsequent
            # request/replay, including serialized multi-threaded callers.
            return _clone_tree(self.static_output)
        finally:
            self._run_lock.release()
