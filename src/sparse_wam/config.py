"""One authoritative configuration for paper equations (1)-(7) and Table 5."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class PaperConfig:
    backbone: Literal["edge", "nano"]
    execution: Literal["eager", "compile", "compile-graph"] = "eager"

    def __post_init__(self):
        if self.backbone not in ("edge", "nano"):
            raise ValueError("Only Cosmos3 Edge and Nano are supported")
        if self.execution not in ("eager", "compile", "compile-graph"):
            raise ValueError("Unsupported execution backend")

    @property
    def layers(self):
        return 28 if self.backbone == "edge" else 36

    @property
    def hidden_size(self):
        return 2048 if self.backbone == "edge" else 4096

    @property
    def spatial_tokens(self):
        # Nano's grid depends on the checkpoint/preprocessing, not its width.
        # Resolve it from the real packed layout once a request is available.
        return 340 if self.backbone == "edge" else None

    @property
    def checkpoint(self):
        tier = "Edge" if self.backbone == "edge" else "Nano"
        return f"nvidia/Cosmos3-{tier}-Policy-DROID"

    def to_dict(self):
        return {
            **asdict(self),
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "future_frames": 8,
            "action_horizon": 32,
            "spatial_tokens": self.spatial_tokens,
            "core_tokens": 80,
            "stable_tokens": 104,
            "retained_tokens": 184,
            "informative_layers": 6,
            "action_weights": [1 / 6, 1 / 3, 1 / 3, 1 / 6],
            "cv_penalty": 1.0,
            "num_steps": 4,
            "guidance": 3.0,
            "shift": 5.0,
            "profile": "step0_conditional",
            "dense_forwards": 1,
            "sparse_forwards": 7,
            "scorer": "main_attention_lse",
            "background_update": "step0_velocity_cache",
            "step0_background_cfg": "unconditional_equals_conditional",
            "mask_scope": "one_action_chunk",
            "checkpoint": self.checkpoint,
        }

    def run_metadata(self, mode, *, spatial_tokens=None):
        """Describe the selected execution, including the native dense control."""
        if mode not in ("dense", "sparse"):
            raise ValueError("mode must be dense or sparse")
        result = {**self.to_dict(), "mode": mode, "sparse_enabled": mode == "sparse"}
        if spatial_tokens is not None:
            result["spatial_tokens"] = int(spatial_tokens)
        if mode == "dense":
            result.update(
                dense_forwards=8,
                sparse_forwards=0,
                profile=None,
                scorer=None,
                core_tokens=None,
                stable_tokens=None,
                retained_tokens=result["spatial_tokens"],
                informative_layers=None,
                action_weights=None,
                cv_penalty=None,
                background_update="current_cfg_velocity",
                step0_background_cfg="native",
                mask_scope=None,
                execution="eager" if self.backbone == "nano" else self.execution,
            )
        return result

    def validate_generation(self, kwargs):
        expected = {"num_steps": 4, "guidance": 3.0, "shift": 5.0}
        for key, value in expected.items():
            if kwargs.get(key, value) != value:
                raise ValueError(f"Paper mode requires {key}={value}")
        for key in ("guidance_interval", "velocity_postprocess_builder", "sampler", "net"):
            if kwargs.get(key) is not None:
                raise ValueError(f"Paper mode cannot combine with a custom {key}")
        if kwargs.get("normalize_cfg", False):
            raise ValueError("Paper mode requires unnormalized CFG")
