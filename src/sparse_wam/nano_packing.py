"""Request-local reuse of Nano denoising metadata, never noisy token values.

Adapted from Piper's _reuse_denoise_packed_sequences. Unlike the broad service
wrapper, this only intercepts packing inside the pinned native _get_velocity:
that caller always refreshes vision/action tokens after packing. Preparation,
multi-sample/per-frame schedules, sound and multi-item layouts use native code.
"""

from contextlib import contextmanager

import torch


@contextmanager
def reuse_nano_packing(model, enabled=True):
    stats = dict(enabled=enabled, calls=0, builds=0, hits=0, bypasses=0)
    if not enabled:
        yield stats
        return
    names = ("_pack_input_sequence", "_get_velocity")
    originals = {name: getattr(model, name) for name in names}
    instance_values = {name: vars(model)[name] for name in names if name in vars(model)}
    cache, owners = {}, []
    active = None

    def velocity(*args, **kwargs):
        nonlocal active
        previous = active
        clean = kwargs.get("gen_data_clean")
        plans = kwargs.get("sequence_plans")
        noise = kwargs.get("noise_x")
        eligible = (
            not args
            and clean is not None
            and getattr(clean, "batch_size", None) == 1
            and getattr(clean, "num_vision_items_per_sample", None) is None
            and isinstance(plans, (tuple, list))
            and len(plans) == 1
            and not getattr(plans[0], "has_sound", False)
            and isinstance(noise, (tuple, list))
            and len(noise) == 1
            and not getattr(model.config, "video_temporal_causal", False)
            and not getattr(model.config, "sound_gen", False)
        )
        active = (id(clean), tuple(id(plan) for plan in plans)) if eligible else None
        if eligible:
            owners.append((clean, plans))  # prevent id reuse for this request
        try:
            return originals["_get_velocity"](*args, **kwargs)
        finally:
            active = previous

    def pack(
        sequence_plans,
        input_text_indexes,
        gen_data_clean,
        input_timesteps,
        include_end_of_generation_token=False,
        skip_text_tokens=False,
        initial_mrope_temporal_offset=0,
    ):
        stats["calls"] += 1
        eligible = (
            active is not None
            and isinstance(input_timesteps, torch.Tensor)
            and input_timesteps.device.type == "cpu"
            and input_timesteps.numel() == 1
            and len(input_text_indexes) == 1
        )
        key = None
        if eligible:
            geometry = tuple(
                None if values is None else tuple((tuple(t.shape), t.dtype, t.device) for t in values)
                for values in (gen_data_clean.x0_tokens_vision, gen_data_clean.x0_tokens_action)
            )
            key = (
                active,
                tuple(tuple(row) for row in input_text_indexes),
                bool(include_end_of_generation_token),
                bool(skip_text_tokens),
                initial_mrope_temporal_offset,
                geometry,
            )
        if key is None or key not in cache:
            result = originals["_pack_input_sequence"](
                sequence_plans,
                input_text_indexes,
                gen_data_clean,
                input_timesteps,
                include_end_of_generation_token=include_end_of_generation_token,
                skip_text_tokens=skip_text_tokens,
                initial_mrope_temporal_offset=initial_mrope_temporal_offset,
            )
            stats["builds"] += 1
            if key is None:
                stats["bypasses"] += 1
            else:
                cache[key] = result
            return result
        result = cache[key]
        # These tensors describe ONLY noisy positions; condition positions are
        # excluded by the native packer. Never alter masks or original RoPE IDs.
        timestep = input_timesteps.item()
        for modality in (result.vision, result.action):
            if modality is not None and modality.timesteps is not None:
                modality.timesteps.fill_(timestep)
        stats["hits"] += 1
        # Native _get_velocity now assigns the CURRENT noisy vision/action
        # tensors and domain IDs, then calls to_cuda() before network execution.
        return result

    model._pack_input_sequence = pack
    model._get_velocity = velocity
    try:
        yield stats
    finally:
        for name in names:
            if name in instance_values:
                setattr(model, name, instance_values[name])
            else:
                delattr(model, name)
        cache.clear()
        owners.clear()
