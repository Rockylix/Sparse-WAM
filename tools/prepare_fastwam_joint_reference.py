"""Export a real LIBERO observation and eager goldens using the original Joint checkout.

Run in the reference checkout's Qi/LIBERO environment. The exported NPZ files can
then be used without that environment, the simulator, or the reference source.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--wan-base", required=True, type=Path,
                        help="Directory containing Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-stats", type=Path)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fastwam_joint/smoke"))
    args = parser.parse_args()
    root = args.reference_root.resolve()
    base = args.wan_base.resolve()
    output = args.output_dir.resolve()
    checkpoint = (args.checkpoint or root / "checkpoints/step_040000.pt").resolve()
    stats = (args.dataset_stats or root / "checkpoints/dataset_stats.json").resolve()
    vae = base / "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    cache = root / "data/text_embeds_cache/libero"
    for path in (checkpoint, stats, vae):
        if not path.is_file():
            parser.error(f"Missing reference asset: {path}")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(base)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    sys.path[:0] = [str(root / "src"), str(root / "tests/libero")]
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import eval_libero_single as native

    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="sim_libero", overrides=["task=libero_joint_2cam224_1e-4"])
    cfg.ckpt = str(checkpoint)
    cfg.seed = args.seed
    cfg.model.redirect_common_files = False
    cfg.EVALUATION.text_cache.directory = str(cache)
    native.set_global_seed(args.seed)
    suite = native.benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    initial = native._get_task_init_states_compat(suite, args.task_id)[0]
    env, task_description = native.get_libero_env(task, native.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(5):
            obs, _, _, _ = env.step(native.get_libero_dummy_action())
        observation = {key: obs[key].copy() for key in (
            "agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
            "robot0_eef_quat", "robot0_gripper_qpos",
        )}
    finally:
        env.close()
    observation["task_description"] = task_description
    condition = native.load_libero_text_condition(cfg, task_description, "cuda", torch.bfloat16)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    native._load_model_checkpoint(model, str(checkpoint))
    model = model.to("cuda").eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(native.load_dataset_stats_from_json(str(stats)))
    height, width = cfg.data.train.video_size
    image, proprio, _ = native._obs_to_model_input(observation, cfg, processor, width, height,
                                                   "cuda", model.torch_dtype)
    infer = dict(condition, input_image=image, proprio=proprio, num_video_frames=native._get_num_video_frames(cfg),
                 action_horizon=32, text_cfg_scale=1.0, num_inference_steps=10, seed=args.seed,
                 rand_device="cpu", expert_cache=False)
    with torch.no_grad():
        raw_dense = model.infer_action(**infer)["action"].clone()
        repeat_dense = model.infer_action(**infer)["action"]
        controller = native.FastWAMJointVersion1Controller(
            mot=model.mot, num_steps=10, top_block_count=6, core_token_count=19,
            stable_token_count=37, velocity_cache=True, torch_compile=False, cuda_graph=False,
        )
        with controller:
            raw_sparse = model.infer_action(**infer)["action"]
    summary = controller.summary()
    if summary["fused_step0_profile_error"] or not summary["fused_step0_profile_used"]:
        raise RuntimeError(f"Native Step0 profile failed: {summary['fused_step0_profile_error']}")
    golden = {"image": image.float().cpu().numpy(), "proprio": proprio.float().cpu().numpy(),
              "core_masks": controller.plan.core_masks.cpu().numpy(),
              "stable_mask": controller.plan.stable_mask.cpu().numpy(),
              "selected_blocks": controller.plan.selected_blocks_ranked.cpu().numpy()}
    for name, raw in (("dense", raw_dense), ("sparse", raw_sparse)):
        action = native._denormalize_action(raw, processor)[0]
        action[..., -1] = action[..., -1] * 2 - 1
        action = native.invert_gripper_action(action)
        action[..., -1] = np.sign(action[..., -1])
        if action.shape != (32, 7) or not np.isfinite(action).all():
            raise ValueError(f"Invalid native {name} action")
        golden[f"{name}_action"] = action
        golden[f"{name}_normalized"] = raw.detach().float().cpu().numpy()
    formatted = native.DEFAULT_PROMPT.format(task=task_description)
    text_key = hashlib.sha256(formatted.encode()).hexdigest()
    text_file = cache / f"{text_key}.t5_len128.wan22ti2v5b.pt"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        tmp = Path(temp)
        np.savez_compressed(tmp / "observation.npz", **observation)
        np.savez_compressed(tmp / "reference_actions.npz", **golden)
        manifest = {
            "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "task_suite": "libero_spatial", "task_id": args.task_id, "seed": args.seed, "initial_state": 0,
            "dummy_steps": 5, "prompt": task_description, "summary": summary,
            "dense_repeat_max_abs_normalized": float((raw_dense - repeat_dense).abs().max().item()),
            "assets_sha256": {p.name: sha256(p) for p in (checkpoint, stats, vae, text_file)},
            "observation_sha256": sha256(tmp / "observation.npz"),
            "reference_actions_sha256": sha256(tmp / "reference_actions.npz"),
        }
        (tmp / "observation.provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in ("observation.npz", "reference_actions.npz", "observation.provenance.json"):
            (tmp / name).replace(output / name)
    print(f"Exported LIBERO observation and Joint dense/K56 eager references to {output}")


if __name__ == "__main__":
    main()
