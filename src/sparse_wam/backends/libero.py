"""Two-camera LIBERO observations and actions for the Joint checkpoint."""

import hashlib
import json
import math

import numpy as np
from PIL import Image
import torch


PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _minmax_parameters(stats, field, dimension):
    value = stats[field]["default"]
    low = torch.tensor(value["global_min"], dtype=torch.float32)
    high = torch.tensor(value["global_max"], dtype=torch.float32)
    if (low.shape != (dimension,) or high.shape != low.shape
            or not torch.isfinite(low).all() or not torch.isfinite(high).all() or (high < low).any()):
        raise ValueError(f"Invalid LIBERO min/max statistics for {field}")
    width = high - low
    constant = width < 1e-4
    scale = 2.0 / torch.where(constant, 2.0, width)
    offset = torch.where(constant, -low, -1.0 - scale * low)
    return scale, offset


def _camera_rgb(raw):
    image = np.asarray(raw)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or 0 in image.shape:
        raise ValueError("LIBERO cameras require nonempty uint8 HWC RGB images")
    rotated = Image.fromarray(np.ascontiguousarray(image[::-1, ::-1]))
    w, h = rotated.size
    ratio = max(224 / w, 224 / h)
    resized = rotated.resize((round(w * ratio), round(h * ratio)), Image.Resampling.BILINEAR)
    left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
    return np.asarray(resized.crop((left, top, left + 224, top + 224)))


def _state_vector(observation, key, size):
    vector = np.asarray(observation[key])
    if vector.shape != (size,) or not np.issubdtype(vector.dtype, np.number) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid LIBERO {key}: expected {size} finite coordinates")
    return vector


class LiberoProcessor:
    def __init__(self, stats_path, text_cache_dir, device, dtype):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        self.state_scale, self.state_offset = _minmax_parameters(stats, "state", 8)
        self.action_scale, self.action_offset = _minmax_parameters(stats, "action", 7)
        self.text_cache_dir = text_cache_dir
        self.device, self.dtype = device, dtype
        self._cached = (None, None)

    def text_condition(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A nonempty LIBERO task description is required")
        cached_instruction, cached_text = self._cached
        if instruction == cached_instruction:
            return cached_text
        formatted = PROMPT.format(task=instruction)
        digest = hashlib.sha256(formatted.encode("utf-8")).hexdigest()
        path = self.text_cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"], payload["mask"]
        if (context.shape != (128, 4096) or mask.shape != (128,)
                or payload.get("prompt", formatted) != formatted):
            raise ValueError(f"Incompatible LIBERO text condition in {path}")
        # The native online encoder pads its context with zeros and attends to all 128 rows.
        context = context.masked_fill(~mask.bool()[:, None], 0)
        if not torch.isfinite(context).all():
            raise ValueError(f"Nonfinite LIBERO text condition in {path}")
        text = (
            context.to(self.device, self.dtype).unsqueeze(0),
            torch.ones((1, 128), dtype=torch.bool, device=self.device),
        )
        self._cached = (instruction, text)
        return text

    def prepare(self, observation):
        cameras = [_camera_rgb(observation[key]) for key in ("agentview_image", "robot0_eye_in_hand_image")]
        image = torch.from_numpy(np.concatenate(cameras, axis=1)).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device, self.dtype) * (2.0 / 255.0) - 1.0
        position = _state_vector(observation, "robot0_eef_pos", 3)
        quat = _state_vector(observation, "robot0_eef_quat", 4)
        gripper = _state_vector(observation, "robot0_gripper_qpos", 2)
        w = np.clip(quat[3], -1.0, 1.0)
        den = np.sqrt(1.0 - w * w)
        axis_angle = np.zeros(3) if math.isclose(den, 0.0) else (quat[:3] * 2.0 * math.acos(w)) / den
        state = torch.tensor(np.concatenate((position, axis_angle, gripper)), dtype=torch.float32)
        state = (state * self.state_scale + self.state_offset).clamp(-5.0, 5.0).unsqueeze(0)
        context, mask = self.text_condition(observation.get("prompt", observation.get("task_description")))
        return dict(input_image=image, proprio=state, prompt=None, context=context, context_mask=mask)

    def environment_action(self, prediction):
        action = ((prediction.detach().float().cpu() - self.action_offset) / self.action_scale).numpy()
        action[..., -1] = np.sign(-(action[..., -1] * 2.0 - 1.0))
        return action
