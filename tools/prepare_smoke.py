"""Download pinned official Edge assets and prepare one recorded DROID observation.

No simulator or robot is required. Policy weights and prepared inputs stay in
Git-ignored directories. The native WAN VAE uses the standard Hugging Face cache,
shared with the vendored runtime; provenance records immutable upstream revisions.
"""

import argparse
import ast
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

MODEL_REPOSITORY = "nvidia/Cosmos3-Edge-Policy-DROID"
MODEL_REVISION = "a74e848cdfd222ec1ff80bd82d4b07eeb3c78d14"
VAE_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
VAE_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
VAE_FILENAME = "Wan2.2_VAE.pth"
SAMPLE_REPOSITORY = "NVIDIA/cosmos-framework"
SAMPLE_REVISION = "e40831fa068fd171894b6cfb269308b441e80a85"
SAMPLE_PATH = ".agents/skills/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example"
EXAMPLE_PATH = ".agents/skills/cookbooks/cosmos3/nim/examples/policy_droid.py"
CAMERAS = ("wrist_image_left", "exterior_image_1_left", "exterior_image_2_left")
PROMPT = "Remove the crumpled paper from the sink and place it in the trash can."
JOINT_POSITION = [
    0.04376881942152977, -0.40943676233291626, 0.42356184124946594,
    -2.431333065032959, -0.027579963207244873, 1.9793097972869873,
    0.6358067393302917,
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    """Cache complete downloads; never leave a partial file at its final name."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Sparse-WAM-smoke"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def compose_observation(frames):
    """Use the exact wrist/top, two exterior/bottom layout of the official example."""
    from PIL import Image

    if len(frames) != 3 or any(frame.size != (640, 360) for frame in frames):
        raise ValueError("The DROID sample must have three 640x360 camera views")
    image = Image.new("RGB", (640, 540))
    image.paste(frames[0], (0, 0))
    for index, frame in enumerate(frames[1:]):
        image.paste(frame.resize((320, 180), Image.Resampling.BILINEAR), (320 * index, 360))
    return np.asarray(image, dtype=np.uint8)


def read_example_metadata(source):
    """Read only literal prompt/state data from the pinned example; never execute it."""
    tree = ast.parse(source)
    states, prompts = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "observation":
                    states.append(ast.literal_eval(value))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--prompt"
        ):
            prompts.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default")
    if len(states) != 1 or len(prompts) != 1:
        raise ValueError("Official example must contain exactly one literal state and default prompt")
    prompt, state = prompts[0], states[0]
    joints = np.asarray(state["observation/joint_position"], dtype=np.float32)
    gripper = float(state["observation/gripper_position"])
    if not isinstance(prompt, str) or not prompt.strip() or joints.shape != (7,):
        raise ValueError("Official example has an invalid prompt or joint state")
    if not np.isfinite(joints).all() or not np.isfinite(gripper):
        raise ValueError("Official example state must be finite")
    return prompt, joints, gripper


def save_observation(frames, destination, *, prompt=PROMPT, joint_position=JOINT_POSITION, gripper_position=0.0):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        np.savez_compressed(
            target,
            **{
                "observation/image": compose_observation(frames),
                "observation/joint_position": np.asarray(joint_position, dtype=np.float32),
                "observation/gripper_position": np.asarray(gripper_position, dtype=np.float32),
                "prompt": np.asarray(prompt),
            },
        )


def prepare_observation(destination, revision):
    import av

    # Resolve a caller-supplied branch/tag once; every subsequent URL uses its SHA.
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        url = f"https://api.github.com/repos/{SAMPLE_REPOSITORY}/commits/{revision}"
        with urllib.request.urlopen(url, timeout=60) as response:
            revision = json.load(response)["sha"]
    base = f"https://raw.githubusercontent.com/{SAMPLE_REPOSITORY}/{revision}"
    cache = destination.parent / "assets" / revision
    example = cache / "policy_droid.py"
    download(f"{base}/{EXAMPLE_PATH}", example)
    prompt, joints, gripper = read_example_metadata(example.read_text(encoding="utf-8"))
    frames, files = [], []
    for camera in CAMERAS:
        remote_path = f"{SAMPLE_PATH}/videos/observation.image.{camera}/chunk-000/file-000.mp4"
        local_path = cache / f"{camera}.mp4"
        download(f"{base}/{remote_path}", local_path)
        if local_path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/"):
            local_path.unlink()
            download(
                f"https://media.githubusercontent.com/media/{SAMPLE_REPOSITORY}/{revision}/{remote_path}",
                local_path,
            )
        with av.open(str(local_path)) as video:
            frames.append(next(video.decode(video=0)).to_image().convert("RGB"))
        files.append({"path": remote_path, "sha256": sha256(local_path)})
    save_observation(frames, destination, prompt=prompt, joint_position=joints, gripper_position=gripper)
    provenance = {
        "repository": SAMPLE_REPOSITORY,
        "resolved_revision": revision,
        "example": EXAMPLE_PATH,
        "example_sha256": sha256(example),
        "frame_index": 0,
        "files": files,
        "prompt": prompt,
        "joint_position": joints.tolist(),
        "gripper_position": gripper,
        "observation_sha256": sha256(destination),
    }
    write_json(destination.with_suffix(".provenance.json"), provenance)
    print(f"Prepared {destination} from {SAMPLE_REPOSITORY}@{revision}", flush=True)


def prepare_checkpoint(destination, revision):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    resolved = HfApi().model_info(MODEL_REPOSITORY, revision=revision).sha
    provenance_path = destination / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("resolved_revision") != resolved:
            raise ValueError("Checkpoint directory belongs to another revision; choose a new --checkpoint-dir")
    snapshot_download(repo_id=MODEL_REPOSITORY, revision=resolved, local_dir=destination, max_workers=4)
    # The bundled vae/ weights are Diffusers format. The native WAN interface
    # needs the original .pth at the exact revision registered by the runtime.
    # Keep this in the Hub cache so its checkpoint downloader reuses the file.
    native_vae = hf_hub_download(
        repo_id=VAE_REPOSITORY, revision=VAE_REVISION, filename=VAE_FILENAME
    )
    write_json(provenance_path, {
        "repository": MODEL_REPOSITORY,
        "resolved_revision": resolved,
        "requested_revision": revision,
        "native_vae": {
            "repository": VAE_REPOSITORY,
            "resolved_revision": VAE_REVISION,
            "filename": VAE_FILENAME,
            "path": str(Path(native_vae).resolve()),
        },
    })
    print(f"Prepared {destination} from {MODEL_REPOSITORY}@{resolved}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/edge"))
    parser.add_argument("--checkpoint-revision", default=MODEL_REVISION)
    parser.add_argument("--observation", type=Path, default=Path("outputs/smoke/observation.npz"))
    parser.add_argument("--sample-revision", default=SAMPLE_REVISION)
    parser.add_argument("--sample-only", action="store_true", help="Prepare the small observation without weights")
    args = parser.parse_args()
    prepare_observation(args.observation, args.sample_revision)
    if not args.sample_only:
        prepare_checkpoint(args.checkpoint_dir, args.checkpoint_revision)


if __name__ == "__main__":
    main()
