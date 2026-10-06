# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# CFG bootstrap and native-sampler wrapping adapted from the pinned Nano k112_sparse helpers.
"""Sparse-WAM velocity caching, applied AFTER CFG and BEFORE the native UniPC update."""

import math

import torch


def expand_future_mask(keep, token_grid, output_grid, patch_size):
    """Map patch tokens to actual velocity pixels, trimming patch padding only."""
    th, tw = token_grid
    height, width = output_grid
    pt, ph, pw = (1, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if min(pt, ph, pw) <= 0:
        raise ValueError("Patch dimensions must be positive")
    if pt != 1 or (th, tw) != (math.ceil(height / ph), math.ceil(width / pw)):
        raise ValueError("Velocity grid is incompatible with the checkpoint patch geometry")
    if tuple(keep.shape) != (8, th * tw):
        raise ValueError("Expected eight frame-specific token masks")
    return keep.reshape(8, th, tw).repeat_interleave(ph, 1).repeat_interleave(pw, 2)[:, :height, :width]


def merge_future(current, cached, keep):
    """Preserve current L0; use current ROI and cached background for L1..L8."""
    if current.shape != cached.shape or current.ndim not in (4, 5):
        raise ValueError("Expected matching [C,9,H,W] or [1,C,9,H,W] predictions")
    if current.shape[-3] != 9 or tuple(keep.shape) != (8, *current.shape[-2:]):
        raise ValueError("Future mask does not match the vision prediction")
    result = current.clone()
    result[..., 1:, :, :] = torch.where(keep, current[..., 1:, :, :], cached[..., 1:, :, :])
    return result


class VelocityCacheSampler:
    """A request-local wrapper; the UniPC solver and its timestep schedule are unchanged."""

    def __init__(self, inner, controller):
        self.inner = inner
        self.controller = controller
        self.steps = 0
        self.cached = None

    def __call__(self, velocity_fn, initial_noise, **kwargs):
        if self.steps or self.cached is not None:
            raise RuntimeError("Velocity cache must not cross action chunks")
        if kwargs.get("num_steps", 4) != 4:
            raise ValueError("Sparse-WAM requires four denoising steps")

        def cached_velocity(noise, timestep):
            current = velocity_fn(noise, timestep)  # native conditional -> unconditional -> CFG
            shape = self.controller.vision_shape
            mask = self.controller.velocity_mask
            if shape is None or mask is None or len(current) != 1:
                raise RuntimeError("Expected one initialized vision/action policy sample")
            count = math.prod(shape)
            flat = current[0]
            if flat.ndim != 1 or flat.numel() <= count:
                raise RuntimeError("Expected flattened [vision | action] velocity")
            vision = flat[:count].reshape(shape)
            if self.steps == 0:
                # Step-0 background uncond has already been replaced by full cond.
                # Consequently this guided cache uses the reliable full prediction there.
                self.cached = vision.detach().clone()
                output = current
            else:
                merged = merge_future(vision, self.cached, mask)
                output = [torch.cat((merged.reshape(-1), flat[count:]))]
            self.steps += 1
            return output

        result = self.inner(cached_velocity, initial_noise, **kwargs)
        if self.steps != 4:
            raise RuntimeError(f"Expected four velocity updates, got {self.steps}")
        return result
