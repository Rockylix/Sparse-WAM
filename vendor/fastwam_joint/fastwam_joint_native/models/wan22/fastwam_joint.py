from typing import Any, Optional

import torch

from logging import getLogger as get_logger

from .fastwam import FastWAM

logger = get_logger(__name__)


class FastWAMJoint(FastWAM):
    """FastWAM variant where action attends to all video latent tokens."""

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError(
                "`video_dit_config` must be provided as dict for FastWAMJoint."
            )
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError(
                "FastWAMJoint requires `video_dit_config['action_conditioned']=false`."
            )
        return super().from_wan22_pretrained(**kwargs)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> full video
        mask[video_seq_len:, :video_seq_len] = True
        return mask

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expert_cache: bool = False,
        expert_cache_reuse_steps: int = 1,
        expert_cache_warmup_steps: int = 1,
        expert_cache_cooldown_steps: int = 1,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if test_action_with_infer_action:
            logger.warning(
                "`FastWAMJoint.infer_joint` ignores `test_action_with_infer_action=True` "
                "and always runs with `test_action_with_infer_action=False`."
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            expert_cache=expert_cache,
            expert_cache_reuse_steps=expert_cache_reuse_steps,
            expert_cache_warmup_steps=expert_cache_warmup_steps,
            expert_cache_cooldown_steps=expert_cache_cooldown_steps,
            test_action_with_infer_action=False,
        )

    @torch.no_grad()
    def _predict_version1_persistent_noise(
        self,
        *,
        packed_latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fixed-shape sparse step without reconstructing full video tokens."""
        controller = getattr(self.mot, "version1_sparse_controller", None)
        if controller is None or not bool(getattr(controller, "persistent_action_only", False)):
            raise RuntimeError("Persistent Version1 inference requires an attached action-only controller")
        if controller.video_grid is None:
            raise RuntimeError("Persistent Version1 inference has no Step0 video grid")
        if getattr(self.mot, "nvtx_profile_label", None):
            self.mot.nvtx_profile_step = int(step_index)
        with self.mot._nvtx_profile_range("sparse_input_embedding"):
            controller.prepare_forward(
                step_index=step_index,
                video_meta={"grid_size": controller.video_grid},
                timestep_video=timestep_video,
                timestep_action=timestep_action,
            )
            (
                embeds_all,
                attention_mask,
                freqs_all,
                context_all,
                t_mod_all,
                video_t,
            ) = controller.prepare_persistent_mot_inputs(
                packed_video_state=packed_latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
            )
        with self.mot._nvtx_profile_range("sparse_transformer"):
            tokens_out = self.mot.forward_version1_sparse(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
                controller=controller,
            )
        with self.mot._nvtx_profile_range("output_heads"):
            pred_video = controller.post_persistent_video(tokens_out["video"], video_t)
            pred_action = self.action_expert.post_dit(tokens_out["action"], {})
        controller.mark_persistent_step_complete()
        return pred_video, pred_action

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        expert_cache: bool = False,
        expert_cache_reuse_steps: int = 1,
        expert_cache_warmup_steps: int = 1,
        expert_cache_cooldown_steps: int = 1,
        action_cache: Any = None,
    ) -> dict[str, Any]:
        # Deployment sets eval once after loading. Walking thousands of child
        # modules again on every chunk costs host time before any CUDA launch.
        if self.training:
            self.eval()
        version1_controller = getattr(self.mot, "version1_sparse_controller", None)
        if action_cache is not None:
            if version1_controller is not None or expert_cache or num_inference_steps != 10 or text_cfg_scale != 1.0 or action_horizon != 32:
                raise ValueError("ActionCache requires native Dense10, cfg=1 and no sparse/expert cache")
            if self._module_torch_compile_enabled or self._graph_mgr is not None:
                raise ValueError("Use the shared original joint MoT execution backend for both methods")
        if version1_controller is not None:
            version1_controller.enable_persistent_action_only()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        with self.mot._nvtx_profile_range("noise_initialization"):
            video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
            latents_video = torch.randn(
                (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            latents_action = torch.randn(
                (1, action_horizon, self.action_expert.action_dim),
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

        with self.mot._nvtx_profile_range("vae_image_encode"):
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
            latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        with self.mot._nvtx_profile_range("text_and_proprio_conditioning"):
            use_prompt = prompt is not None
            use_context = context is not None or context_mask is not None
            if use_prompt and use_context:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            if not use_prompt and not use_context:
                raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

            if use_prompt:
                context, context_mask = self.encode_prompt(prompt)
            else:
                if context is None or context_mask is None:
                    raise ValueError("`context` and `context_mask` must be both provided together.")
                if context.ndim == 2:
                    context = context.unsqueeze(0)
                if context_mask.ndim == 1:
                    context_mask = context_mask.unsqueeze(0)
                if context.ndim != 3 or context_mask.ndim != 2:
                    raise ValueError(
                        f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                    )
                context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if proprio is not None:
                context, context_mask = self._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                )

        with self.mot._nvtx_profile_range("schedule_setup"):
            infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
            infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            action_expert_cache = self._make_action_expert_cache(
                enabled=expert_cache,
                num_inference_steps=num_inference_steps,
                reuse_steps=expert_cache_reuse_steps,
                warmup_steps=expert_cache_warmup_steps,
                cooldown_steps=expert_cache_cooldown_steps,
            )
        packed_latents_video: torch.Tensor | None = None
        for step_idx, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(
            zip(
                infer_timesteps_video,
                infer_deltas_video,
                infer_timesteps_action,
                infer_deltas_action,
            )
        ):
            if action_cache is not None and action_cache.hit and 0 < step_idx < 8:
                continue
            if action_cache is not None:
                if step_idx == 8:
                    if not action_cache.hit:
                        action_cache.stage_step8(latents_video, latents_action)
                action_cache.steps.append(step_idx)
                if step_idx >= 8:
                    action_cache.native_tail.append({"step":step_idx, "video_t":step_t_video.detach(),
                        "action_t":step_t_action.detach(), "video_delta":step_delta_video.detach(),
                        "action_delta":step_delta_action.detach()})
            if getattr(self.mot, "nvtx_profile_label", None):
                self.mot.nvtx_profile_step = step_idx
            with self.mot._nvtx_profile_range("denoise_step"):
                video_dtype = (
                    latents_video.dtype
                    if latents_video is not None
                    else packed_latents_video.dtype
                )
                timestep_video = step_t_video.unsqueeze(0).to(dtype=video_dtype, device=self.device)
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                if version1_controller is None or version1_controller.is_dense_step(step_idx):
                    if latents_video is None:
                        latents_video = version1_controller.full_video_state_for_refresh()
                    with self.mot._nvtx_profile_range("dense_noise_prediction"):
                        pred_video_posi, pred_action_posi = self._predict_joint_noise(
                            latents_video=latents_video,
                            latents_action=latents_action,
                            timestep_video=timestep_video,
                            timestep_action=timestep_action,
                            context=context,
                            context_mask=context_mask,
                            fuse_vae_embedding_in_latents=fuse_flag,
                            gt_action=None,
                            expert_cache_state=action_expert_cache,
                            expert_cache_step=step_idx,
                            action_cache_probe=action_cache if step_idx == 0 else None,
                        )
                    if action_cache is not None and step_idx == 0 and action_cache.hit:
                        latents_video, latents_action = action_cache.restore(first_frame_latents)
                        continue
                    with self.mot._nvtx_profile_range("video_scheduler"):
                        latents_video = self.infer_video_scheduler.step(
                            pred_video_posi, step_delta_video, latents_video
                        )
                        latents_video[:, :, 0:1] = first_frame_latents.clone()
                    if version1_controller is not None:
                        with self.mot._nvtx_profile_range("token_pack_and_cache_init"):
                            packed_latents_video = version1_controller.pack_persistent_video_state(
                                latents_video
                            )
                        latents_video = None
                else:
                    if packed_latents_video is None:
                        raise RuntimeError("Persistent Version1 video patch state is unavailable")
                    pred_video_posi, pred_action_posi = self._predict_version1_persistent_noise(
                        packed_latents_video=packed_latents_video,
                        latents_action=latents_action,
                        timestep_video=timestep_video,
                        timestep_action=timestep_action,
                        step_index=step_idx,
                    )
                    with self.mot._nvtx_profile_range("video_scheduler"):
                        packed_latents_video = self.infer_video_scheduler.step(
                            pred_video_posi,
                            step_delta_video,
                            packed_latents_video,
                        )
                    with self.mot._nvtx_profile_range("background_cache_fill"):
                        version1_controller.update_persistent_video_cache(
                            packed_latents_video, step_delta_video,
                            self.infer_video_scheduler, first_frame_latents,
                        )
                with self.mot._nvtx_profile_range("action_scheduler"):
                    latents_action = self.infer_action_scheduler.step(pred_action_posi, step_delta_action, latents_action)

        if action_cache is not None:
            action_cache.complete()
        self._log_action_expert_cache(action_expert_cache)
        with self.mot._nvtx_profile_range("action_cpu_return"):
            return {
                "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            }
