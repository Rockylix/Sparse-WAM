"""Select a Cosmos backend and attach Sparse-WAM to its native sampler."""

import json
import math
import os
from pathlib import Path
import sys

import torch

from .controller import SparseWAMController
from .nano_packing import reuse_nano_packing
from .velocity import VelocityCacheSampler


def activate_runtime(backbone):
    if backbone not in ("edge", "nano"):
        raise ValueError("Unknown backbone")
    root = Path(__file__).resolve().parents[2]
    runtime = root / "vendor" / backbone
    if not (runtime / "cosmos_framework").is_dir():
        raise RuntimeError(
            "Use an editable install from the complete source checkout; vendored runtime is missing"
        )
    # Upstream defaults to training imports, including optional storage clients.
    # Select inference before framework modules read their process-wide flags.
    os.environ.setdefault("COSMOS_TRAINING", "0")
    existing = sys.modules.get("cosmos_framework")
    if existing is not None:
        paths = [Path(p).resolve() for p in existing.__path__]
        if paths != [(runtime / "cosmos_framework").resolve()]:
            raise RuntimeError("Another cosmos_framework ABI is loaded; run Edge/Nano in separate processes")
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    if backbone == "nano":
        if os.environ.get("COSMOS3_LAYER_OFFLOAD", "0") not in ("0", "", "false", "False"):
            raise RuntimeError("Nano GEN-resident mode conflicts with COSMOS3_LAYER_OFFLOAD")
        # Set before importing the framework: its module globals read these flags.
        for key in ("COSMOS3_REASONER_OFFLOAD", "COSMOS3_REASONER_KVCACHE", "COSMOS3_POLICY_REASONER_CACHE"):
            os.environ[key] = "1"
        os.environ.setdefault("COSMOS3_REASONER_EMPTY_CACHE", "1")
        os.environ.setdefault("COSMOS3_REASONER_PIN_CPU_MASTER", "0")
    return runtime


def normalize_caption(batch, key):
    """Serialize structured task text before the tokenizer, for BOTH dense and sparse modes."""
    captions = batch.get(key)
    if not isinstance(captions, (list, tuple)) or len(captions) != 1:
        raise ValueError("Expected one task caption per policy batch")
    caption = captions[0]
    if isinstance(caption, dict):
        caption = json.dumps(caption, ensure_ascii=False)
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("A nonempty task instruction is required")
    return {**batch, key: [caption]}


class SparseWAMRunner:
    def __init__(self, model, config, *, mode="sparse", generate=None):
        if mode not in ("dense", "sparse"):
            raise ValueError("mode must be dense or sparse")
        self.model, self.config, self.mode = model, config, mode
        self.generate = generate if generate is not None else model.generate_samples_from_batch
        self.last_summary = None
        if model.config.rectified_flow_inference_config.scheduler_type != "unipc":
            raise ValueError("This release supports the native policy UniPC sampler only")
        if type(model.sampler).__name__ != "UniPCSampler":
            raise ValueError("A fixed-step/distilled or replaced sampler is not supported")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("This release supports single-rank policy inference only")

    @torch.inference_mode()
    def __call__(self, data_batch, **kwargs):
        self.config.validate_generation(kwargs)
        if kwargs.get("upsample_task") is not None:
            raise ValueError("Prompt upsampling is outside the policy reproduction protocol")
        arguments = {"guidance": 3.0, "num_steps": 4, "shift": 5.0, **kwargs}
        batch = normalize_caption(data_batch, self.model.input_caption_key)
        self.last_summary = None
        if self.mode == "dense":
            result = self.generate(batch, **arguments)
            summary = self.config.run_metadata("dense")
        else:
            controller = SparseWAMController(self.model.net, self.config)
            sampler = VelocityCacheSampler(self.model.sampler, controller)
            with (
                reuse_nano_packing(
                    self.model, self.config.backbone == "nano" and self.config.nano_packing_cache
                ) as packing_stats,
                controller,
            ):
                result = self.generate(batch, **{**arguments, "sampler": sampler})
            summary = {
                **controller.finish(),
                **self.config.run_metadata("sparse", spatial_tokens=controller.geometry[2]),
                "velocity_updates": sampler.steps,
                "input_packing": dict(packing_stats),
            }
        for name in ("action", "vision"):
            values = result.get(name)
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 1
                or not isinstance(values[0], torch.Tensor)
                or values[0].numel() == 0
                or not torch.isfinite(values[0]).all()
            ):
                raise RuntimeError(f"Missing or nonfinite {name} output")
        if self.mode == "dense":
            patch = getattr(getattr(self.model, "net", None), "latent_patch_size", None)
            if patch is not None:
                ph, pw = (patch, patch) if isinstance(patch, int) else patch[-2:]
                height, width = result["vision"][0].shape[-2:]
                grid = [math.ceil(height / ph), math.ceil(width / pw)]
                summary.update(
                    spatial_tokens=math.prod(grid), retained_tokens=math.prod(grid), token_grid=grid
                )
        self.last_summary = summary
        return result
