# SPDX-FileCopyrightText: 2026 Rockylix
# SPDX-License-Identifier: MIT
"""Native FastWAM-Joint Dense and K56 inference over one LIBERO observation."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch

from ..runtime import activate_runtime
from .libero import LiberoProcessor


@dataclass(frozen=True)
class JointSettings:
    checkpoint_path: Path
    vae_path: Path
    dataset_stats_path: Path
    text_cache_dir: Path
    seed: int


def resolve_settings(config, checkpoint, overrides, native_config, seed):
    allowed = {"checkpoint_path", "vae_path", "dataset_stats_path", "text_cache_dir"}
    extra = overrides.keys() - allowed
    if extra:
        raise ValueError(f"Unsupported FastWAM-Joint native settings: {sorted(extra)}")
    base = Path(native_config).resolve().parent if native_config is not None else Path.cwd()
    paths = {}
    for name, value in overrides.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain a path")
        paths[name] = (base / Path(value).expanduser()).resolve()
    if checkpoint is not None:
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
            raise ValueError("checkpoint must contain a local file path")
        paths["checkpoint_path"] = Path(checkpoint).expanduser().resolve()
    paths.setdefault("checkpoint_path", Path(config.checkpoint).resolve())
    directory = paths["checkpoint_path"].parent
    paths.setdefault("vae_path", directory / "Wan2.2_VAE.pth")
    paths.setdefault("dataset_stats_path", directory / "dataset_stats.json")
    paths.setdefault("text_cache_dir", directory / "text_embeds_cache/libero")
    for name, path in paths.items():
        exists = path.is_dir() if name == "text_cache_dir" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Missing FastWAM-Joint {name}: {path}")
    return JointSettings(**paths, seed=seed)


def _load_model(settings, runtime):
    from omegaconf import OmegaConf
    from fastwam_joint_native.models.wan22.fastwam_joint import FastWAMJoint

    # Resolve the upstream YAML exactly as the libero_joint_2cam224_1e-4 task does.
    definition = OmegaConf.load(runtime / "configs/model/fastwam_joint.yaml")
    definition.mot_checkpoint_mixed_attn = False
    cfg = OmegaConf.create({
        "model": definition,
        "data": {"train": {"processor": {"proprio_output_dim": 8, "action_output_dim": 7}}},
    })
    options = OmegaConf.to_container(cfg.model, resolve=True)
    model = FastWAMJoint.from_wan22_pretrained(
        device="cuda", torch_dtype=torch.bfloat16, vae_path=str(settings.vae_path),
        model_id=options["model_id"], tokenizer_model_id=options["tokenizer_model_id"],
        tokenizer_max_len=options["tokenizer_max_len"], load_text_encoder=False,
        proprio_dim=options["proprio_dim"], redirect_common_files=False,
        video_dit_config=options["video_dit_config"], action_dit_config=options["action_dit_config"],
        skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None,
        mot_checkpoint_mixed_attn=options["mot_checkpoint_mixed_attn"],
        video_train_shift=options["video_scheduler"]["train_shift"],
        video_infer_shift=options["video_scheduler"]["infer_shift"],
        video_num_train_timesteps=options["video_scheduler"]["num_train_timesteps"],
        action_train_shift=options["action_scheduler"]["train_shift"],
        action_infer_shift=options["action_scheduler"]["infer_shift"],
        action_num_train_timesteps=options["action_scheduler"]["num_train_timesteps"],
    )
    weights = torch.load(settings.checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    load_joint_weights(model, weights)
    return model.requires_grad_(False).eval()


def load_joint_weights(model, weights):
    for name in ("mot", "proprio_encoder"):
        if name not in weights:
            raise ValueError(f"FastWAM-Joint checkpoint is missing {name}")
        state = weights[name]
        if name == "mot":
            # The LIBERO checkpoint also carries spectral-plan adapter weights.
            # The reference Joint MoT does not register or evaluate those modules.
            state = {key: value for key, value in state.items() if not key.startswith("plan_adapters.")}
        getattr(model, name).load_state_dict(state, strict=True)


class JointRunner:
    def __init__(self, model, config, mode, seed, controller_type):
        self.model, self.config, self.mode = model, config, mode
        self.seed = seed
        self.controller_type = controller_type
        self.last_summary = None
        self.capture_selection = False
        self._lock = Lock()

    def __call__(self, prepared):
        # Controller attachment, graph buffers and last_summary belong to this model.
        with self._lock:
            self.last_summary = None
            metadata = self.config.run_metadata(self.mode)
            if self.mode == "dense":
                prediction = self.model.infer_action(**prepared)
            else:
                controller = self.controller_type(
                    mot=self.model.mot, num_steps=10, torch_compile=self.config.execution != "eager",
                    cuda_graph=self.config.execution == "compile-graph",
                )
                with controller:
                    prediction = self.model.infer_action(**prepared)
                metadata.update(controller.summary())
                if self.capture_selection:
                    plan = controller.plan
                    metadata["core_positions"] = [mask.nonzero().flatten().tolist() for mask in plan.core_masks]
                    metadata["stable_positions"] = plan.stable_mask.nonzero().flatten().tolist()
                for key in ("fused_step0_profile_error", "torch_compile_fallback_error", "cuda_graph_fallback_error"):
                    if metadata.get(key):
                        raise RuntimeError(f"FastWAM-Joint {key}: {metadata[key]}")
            action = prediction["action"]
            if action.shape != (32, 7) or not torch.isfinite(action).all():
                raise RuntimeError("FastWAM-Joint must return a finite [32,7] action chunk")
            self.last_summary = metadata
            return prediction


class JointService:
    def __init__(self, runner, processor):
        self.model = runner.model
        self.sparse_wam_runner = runner
        self.processor = processor

    def prepare_observation(self, observation):
        return {
            **self.processor.prepare(observation),
            "action_horizon": 32, "num_video_frames": 9, "text_cfg_scale": 1.0,
            "num_inference_steps": 10, "negative_prompt": "", "sigma_shift": None,
            "seed": self.sparse_wam_runner.seed, "rand_device": "cpu", "expert_cache": False,
            "tiled": False,
        }

    def infer_prepared(self, prepared):
        return self.sparse_wam_runner(prepared)

    def infer(self, observation):
        prediction = self.infer_prepared(self.prepare_observation(observation))
        return {"action": self.processor.environment_action(prediction["action"])}


def create_joint_service(config, checkpoint, overrides, native_config, mode, seed):
    settings = resolve_settings(config, checkpoint, overrides, native_config, seed)
    runtime = activate_runtime("fastwam-joint")
    from fastwam_joint_native.models.wan22.fastwam_joint_version1 import FastWAMJointVersion1Controller

    processor = LiberoProcessor(settings.dataset_stats_path, settings.text_cache_dir, "cuda", torch.bfloat16)
    model = _load_model(settings, runtime)
    runner = JointRunner(model, config, mode, settings.seed, FastWAMJointVersion1Controller)
    return JointService(runner, processor), settings
