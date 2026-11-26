"""Authoritative configuration for Sparse-WAM token selection and execution."""

from dataclasses import asdict, dataclass
from typing import Literal, get_args

Backbone = Literal["edge", "nano", "fastwam-joint"]
BACKBONES = get_args(Backbone)


@dataclass(frozen=True)
class SparseWAMConfig:
    backbone: Backbone
    execution: Literal["eager", "compile", "compile-graph"] = "eager"
    nano_packing_cache: bool = True

    def __post_init__(self):
        if self.backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        if self.execution not in ("eager", "compile", "compile-graph"):
            raise ValueError("Unsupported execution backend")

    @property
    def layers(self):
        return {"edge": 28, "nano": 36, "fastwam-joint": 30}[self.backbone]

    @property
    def hidden_size(self):
        return {"edge": 2048, "nano": 4096, "fastwam-joint": 3072}[self.backbone]

    @property
    def spatial_tokens(self):
        # Nano's grid depends on the checkpoint/preprocessing, not its width.
        # Resolve it from the real packed layout once a request is available.
        return {"edge": 340, "nano": None, "fastwam-joint": 98}[self.backbone]

    @property
    def checkpoint(self):
        if self.backbone == "fastwam-joint":
            return "checkpoints/fastwam_joint/step_040000.pt"
        tier = "Edge" if self.backbone == "edge" else "Nano"
        return f"nvidia/Cosmos3-{tier}-Policy-DROID"

    def to_dict(self):
        if self.backbone == "fastwam-joint":
            return {
                "backbone": self.backbone, "execution": self.execution,
                "layers": self.layers, "video_hidden_size": self.hidden_size, "action_hidden_size": 1024,
                "future_frames": 2, "spatial_tokens": self.spatial_tokens, "action_horizon": 32,
                "action_dim": 7, "proprio_dim": 8, "core_tokens": 19, "stable_tokens": 37,
                "retained_tokens": 56, "informative_layers": 6, "cv_penalty": 1.0,
                "action_weights": [1 / 6, 1 / 3, 1 / 3, 1 / 6],
                "num_steps": 10, "guidance": 1.0, "shift": 5.0, "dense_forwards": 1,
                "sparse_forwards": 9, "profile": "step0_joint", "scorer": "main_attention_lse",
                "background_update": "step0_velocity_cache", "mask_scope": "one_action_chunk",
                "checkpoint": self.checkpoint,
            }
        return {
            **asdict(self),
            "nano_packing_cache": self.backbone == "nano" and self.nano_packing_cache,
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
                dense_forwards=10 if self.backbone == "fastwam-joint" else 8,
                sparse_forwards=0,
                profile=None,
                scorer=None,
                core_tokens=None,
                stable_tokens=None,
                retained_tokens=result["spatial_tokens"],
                informative_layers=None,
                action_weights=None,
                cv_penalty=None,
                background_update=None,
                mask_scope=None,
                execution=self.execution if self.backbone == "edge" else "eager",
            )
            if self.backbone in ("edge", "nano"):
                result.update(background_update="current_cfg_velocity", step0_background_cfg="native",
                              nano_packing_cache=False)
        return result

    def validate_generation(self, kwargs):
        joint = self.backbone == "fastwam-joint"
        expected = {"num_steps": 10 if joint else 4, "guidance": 1.0 if joint else 3.0, "shift": 5.0}
        for key, value in expected.items():
            if kwargs.get(key, value) != value:
                raise ValueError(f"Sparse-WAM requires {key}={value}")
        for key in ("guidance_interval", "velocity_postprocess_builder", "sampler", "net"):
            if kwargs.get(key) is not None:
                raise ValueError(f"Sparse-WAM cannot combine with a custom {key}")
        if kwargs.get("normalize_cfg", False):
            raise ValueError("Sparse-WAM requires unnormalized CFG")
