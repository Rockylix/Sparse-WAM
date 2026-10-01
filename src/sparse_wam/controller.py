# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Lifecycle and native hook ABI derived from the two pinned method branches.
"""One conditional dense forward, seven sparse forwards, one fresh plan per chunk."""

import torch

from .backends import make_backend
from .layout import _action_query_layout
from .packing import NativePacking, selected_positions
from .scoring import profile_geometry
from .selection import select_tokens
from .velocity import expand_future_mask, merge_future


class SparseWAMController:
    def __init__(self, net, config, *, backend=None, packing=None):
        self.net, self.config = net, config
        self.model = net.language_model.model
        self.layers = list(self.model.layers)
        if len(self.layers) != config.layers:
            raise ValueError(f"{config.backbone} requires {config.layers} layers")
        for layer in self.layers:
            if int(layer.self_attn.hidden_size) != config.hidden_size:
                raise ValueError("Checkpoint hidden size does not match the selected backbone")
        self.backend = backend if backend is not None else make_backend(config)
        self.packing = packing if packing is not None else NativePacking()
        self.handles = []
        self.current = None
        self.layout = None
        self.geometry = None
        self.layout_cache = {}
        self.plan = None
        self.profiles = []
        self.selected = None
        self.vision_shape = None
        self.velocity_mask = None
        self.conditional_vision = None
        self.forward_count = 0
        self.completed = 0
        self.block_calls = 0
        self.pack_count = 0
        self.restore_count = 0
        self.stack_active = False
        self.entered = False
        self.backend_started = False
        self.last_block = -1

    @property
    def profiling(self):
        return self.current == (0, "conditional")

    def __enter__(self):
        if self.entered or self.forward_count:
            raise RuntimeError("A controller cannot be reused across requests")
        for name in ("_robolab_version1_controller", "_k112_sparse_controller"):
            if getattr(self.model, name, None) is not None:
                raise RuntimeError("Another sparse controller already owns this model")
        try:
            self.backend_started = True
            self.backend.enter(self.layers)
            setattr(self.model, self.backend.hook_name, self)
            self.handles.append(self.net.register_forward_pre_hook(self._pre, with_kwargs=True))
            self.handles.append(
                self.net.register_forward_hook(self._post, with_kwargs=True, always_call=True)
            )
            self.entered = True
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        self.abort_stack()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if getattr(self.model, self.backend.hook_name, None) is self:
            delattr(self.model, self.backend.hook_name)
        if self.backend_started:
            self.backend.exit()
        self.backend_started = False
        self.current = None
        self.entered = False

    def _pre(self, module, args, kwargs):
        if kwargs.get("und_only", args[2] if len(args) > 2 else False):
            self.current = None
            return
        if self.forward_count >= 8:
            raise RuntimeError("More than eight CFG forwards; unexpected schedule")
        packed = args[0] if args else kwargs["packed_seq"]
        entry = self.layout_cache.get(id(packed))
        if entry is None:
            layout = _action_query_layout(torch, packed)
            self.layout_cache[id(packed)] = (packed, layout)
        else:
            layout = entry[1]
        if self.layout is None:
            self.layout = layout
            self.geometry = profile_geometry(layout)
            if self.geometry[2] != self.config.spatial_tokens:
                raise ValueError("Policy layout differs from the paper's backbone configuration")
        elif layout != self.layout:
            raise ValueError("GEN/action layout changed within one action chunk")
        self.current = (
            self.forward_count // 2,
            "conditional" if self.forward_count % 2 == 0 else "unconditional",
        )
        self.forward_count += 1

    def begin_stack(self, *, hidden_states, position_embeddings, natten_metadata_list, memory_gen_only=None):
        if self.current is None:
            return  # native UND-only prefill
        if self.stack_active or natten_metadata_list is not None or hidden_states.get("is_sharded", False):
            raise ValueError("Nested, NATTEN, and context-parallel stacks are unsupported")
        if int(hidden_states["_num_full_tokens"]) != self.layout["num_gen_tokens"]:
            raise ValueError("A stack must begin with the original GEN layout")
        self.original = hidden_states
        self.rope = position_embeddings
        self.side_buffer = None if self.profiling else self.packing.gen(hidden_states).detach().clone()
        self.stack_active = True
        self.last_block = -1

    def run_layer(self, *, block, decoder_layer, hidden_states, attention_mask, memory_value, gen_only):
        if not self.stack_active or block != self.last_block + 1:
            raise RuntimeError("Unexpected sparse Transformer layer order")
        self.last_block = block
        if not self.profiling and block == 0:
            if self.plan is None:
                raise RuntimeError("Unconditional sparsity requires the conditional full-step plan")
            if self.selected is None:
                self.selected = selected_positions(self.layout, self.plan.keep)
            hidden_states, self.rope = self.packing.slice(hidden_states, self.rope, self.selected)
            self.pack_count += 1
        output, metadata, kv, scores = self.backend.run(
            decoder_layer,
            hidden_states,
            attention_mask,
            self.rope,
            memory_value,
            gen_only,
            self.geometry,
            self.profiling,
        )
        if self.profiling:
            if scores is None:
                raise RuntimeError("Main attention did not return a relevance profile")
            self.profiles.append(scores.detach().clone())
        elif self.packing.gen(output).shape[0] != self.selected.numel():
            raise RuntimeError("Sparse layer changed the retained token count")
        self.block_calls += 1
        return output, metadata, kv

    def end_stack(self, hidden_states):
        if not self.stack_active or self.last_block != self.config.layers - 1:
            raise RuntimeError("Incomplete Transformer stack")
        if self.profiling:
            self.plan = select_tokens(torch.stack(self.profiles))
            self.profiles.clear()
            restored = hidden_states
        else:
            gen = self.side_buffer.index_copy(0, self.selected, self.packing.gen(hidden_states))
            restored = self.packing.replace_gen(self.original, gen)
            self.restore_count += 1
        self.completed += 1
        self.abort_stack()
        return restored

    def abort_stack(self):
        self.stack_active = False
        self.original = None
        self.rope = None
        self.side_buffer = None

    def _post(self, module, args, kwargs, output):
        current = self.current
        try:
            if output is None or current is None:
                return None  # do not mask an exception raised by the native forward
            if self.stack_active:
                raise RuntimeError("Full GEN restoration must precede output heads")
            predictions = output.get("preds_vision")
            if not predictions or len(predictions) != 1:
                raise ValueError("Paper adapter requires one vision/action sample")
            prediction = predictions[0]
            shape = tuple(prediction.shape)
            if self.vision_shape is None:
                if prediction.ndim not in (4, 5) or prediction.shape[-3] != 9:
                    raise ValueError("Expected L0 plus eight future velocity frames")
                self.vision_shape = shape
                self.velocity_mask = expand_future_mask(
                    self.plan.keep,
                    self.layout["latent_shape_thw"][1:],
                    shape[-2:],
                    self.net.latent_patch_size,
                )
            elif shape != self.vision_shape:
                raise ValueError("Vision prediction shape changed during the request")
            if current == (0, "conditional"):
                self.conditional_vision = prediction.detach().clone()
            elif current == (0, "unconditional"):
                # Only conditional was full. Do not bootstrap omitted background
                # from incomplete uncond hidden states: delta_CFG=0 there at step 0.
                corrected = merge_future(prediction, self.conditional_vision, self.velocity_mask)
                result = dict(output)
                result["preds_vision"] = [corrected]
                self.conditional_vision = None
                return result
            return None
        finally:
            self.current = None

    def finish(self):
        expected = (8, 8, 8 * self.config.layers, 7, 7)
        actual = (self.forward_count, self.completed, self.block_calls, self.pack_count, self.restore_count)
        if self.stack_active or self.plan is None or actual != expected:
            raise RuntimeError(f"Incomplete Sparse-WAM execution: {actual}, expected {expected}")
        return {
            **self.config.to_dict(),
            "completed_forwards": self.completed,
            "block_calls": self.block_calls,
            "pack_count": self.pack_count,
            "restore_count": self.restore_count,
            "selected_layers": self.plan.layers.tolist(),
        }
