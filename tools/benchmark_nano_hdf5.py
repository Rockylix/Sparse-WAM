"""Piper HDF5 input adapter for the unchanged Sparse-WAM Nano runner.

Only dataset reading/batch conversion comes from Piper. Model, attention,
selection, compilation and velocity caching come from this source checkout.
No robot connection, model downloads, or VAE decoding.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from sparse_wam.benchmark_execution import ReplayCounter
from sparse_wam.config import SparseWAMConfig
from sparse_wam.runtime import SparseWAMRunner, activate_runtime, normalize_caption


def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(v) for v in value)
    return copy.deepcopy(value)


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(x):
        if torch.is_tensor(x):
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                digest.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        else:
            digest.update(repr(x).encode())

    visit(value)
    return digest.hexdigest()


def metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    delta = b - a
    return dict(
        mse=float(delta.square().mean()),
        relative_l2=float(delta.norm() / a.norm().clamp_min(1e-12)),
        cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        max_absolute_error=float(delta.abs().max()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--piper-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hdf-root", type=Path, required=True)
    p.add_argument("--dataset-config", type=Path, required=True)
    p.add_argument("--vae-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument("--sample-index", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        p.error("positive warmup/repeats required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    runtime = activate_runtime("nano")
    sys.path.append(str(args.piper_root.resolve()))
    import cosmos_framework
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs
    from piper_cosmos.cosmos3.piper14_hdf5_action_dataset import Piper14HDF5ActionDataset
    from piper_cosmos.deployment.cosmos_piper14_policy import build_data_batch_from_sample

    assert Path(cosmos_framework.__file__).resolve().parent == runtime / "cosmos_framework"
    payload = json.loads((args.checkpoint / "config.json").read_text())
    native = payload["model"]["config"]
    native["compile"].update(enabled=False, use_cuda_graphs=False)
    # Preserve the .pth suffix: HF snapshots may symlink to extensionless blobs.
    native["tokenizer"]["vae_path"] = str(args.vae_path.absolute())
    native["vlm_config"]["tokenizer"]["pretrained_model_name"] = str(args.tokenizer_path.resolve())
    config_path = args.output_dir / "native_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n")

    class LocalArgs(ActionServerArgs):
        def build_setup_overrides(self):
            setup = super().build_setup_overrides()
            setup.dp_replicate_size = setup.dp_shard_size = setup.cp_size = setup.cfgp_size = 1
            setup.use_torch_compile = setup.use_cuda_graphs = setup.compile_dynamic = False
            setup.guardrails = setup.offload_guardrail_models = False
            return setup

    service = ActionModelService(LocalArgs(
        checkpoint=CheckpointOverrides(checkpoint_path=str(args.checkpoint.resolve()), config_file=str(config_path.resolve())),
        output_dir=args.output_dir, action_chunk_size=32, raw_action_dim=14, max_action_dim=64,
        seed=args.seed, guidance=3.0, num_steps=4, fps=30, format_prompt_as_json=True,
    ))
    model = service.model
    model.net.pad_for_cuda_graphs = False
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.accumulated_recompile_limit = 4096
    raw = Piper14HDF5ActionDataset(
        root=str(args.hdf_root.resolve()), config_path=str(args.dataset_config.resolve()),
        fps=30.0, chunk_length=32, mode="policy", use_state=True, viewpoint="concat_view", stride=1,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=None, cfg_dropout_rate=0.0, max_action_dim=64,
        append_viewpoint_info=True, append_duration_fps_timestamps=True,
        append_resolution_info=True, append_idle_frames=False, format_prompt_as_json=True,
    )
    sample = ActionSFTDataset(raw, transform, "480")[args.sample_index]
    batch = normalize_caption(build_data_batch_from_sample(sample), model.input_caption_key)
    batch_hash = fingerprint(batch)
    config = SparseWAMConfig("nano", "compile-graph")
    capture = {}
    generate = model.generate_samples_from_batch

    def timed_generate(*a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = generate(*a, **kw)
        torch.cuda.synchronize()
        capture["generation_s"] = time.perf_counter() - start
        return result

    runner = SparseWAMRunner(model, config, generate=timed_generate)
    prefills = [0]

    def pre_hook(module, a, kw):
        if kw.get("und_only", a[2] if len(a) > 2 else False):
            prefills[0] += 1

    handle = model.net.register_forward_pre_hook(pre_hook, with_kwargs=True)
    rows, errors, references = [], [], {}
    source = raw.index[args.sample_index]
    report = dict(
        status="running", checkpoint=str(args.checkpoint.resolve()), hdf_root=str(args.hdf_root.resolve()),
        hdf_episode=str(source.episode_path), hdf_frame=source.t, sample_index=args.sample_index,
        seed=args.seed, warmup=args.warmup, repeats=args.repeats, batch_sha256=batch_hash,
        caption=batch[model.input_caption_key], config=config.to_dict(),
        runtime=str(cosmos_framework.__file__), python=sys.executable, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), argv=sys.argv,
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        timing_scope="synchronized native generate_samples_from_batch; input clone/HDF preprocessing and runner validation outside primary clock; no VAE decode; warm instruction cache",
        offload={k: os.environ.get(k) for k in ("COSMOS3_REASONER_OFFLOAD", "COSMOS3_REASONER_KVCACHE", "COSMOS3_LAYER_OFFLOAD")},
        rows=rows, paired_errors=errors,
    )

    def save():
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for round_id in range(args.warmup + args.repeats):
            outputs = {}
            for mode in (("dense", "sparse") if round_id % 2 == 0 else ("sparse", "dense")):
                prepared = clone(batch)
                assert fingerprint(prepared) == batch_hash
                runner.mode = mode
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with ReplayCounter() as replay:
                    result = runner(prepared, guidance=3.0, seed=[args.seed], num_steps=4, shift=5.0, has_negative_prompt=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                outputs[mode] = {k: result[k][0].detach().cpu().clone() for k in ("action", "vision")}
                measured = round_id >= args.warmup
                row = dict(round=round_id, mode=mode, measured=measured,
                           generation_s=capture["generation_s"], runner_s=elapsed,
                           graph_replays=replay.count, prefill_calls=prefills[0], summary=runner.last_summary)
                # The actual checkpoint is an override of the DROID default.
                row["summary"] = {**row["summary"], "checkpoint": str(args.checkpoint.resolve())}
                rows.append(row)
                assert prefills[0] == 2, "Reasoner KV cache missed"
                if measured:
                    assert replay.count == (0 if mode == "dense" else 288)
                    if mode not in references:
                        references[mode] = clone(outputs[mode])
                    row["repeat_exact"] = {k: torch.equal(references[mode][k], outputs[mode][k]) for k in outputs[mode]}
                    assert all(row["repeat_exact"].values()), "Repeated fixed-input output changed"
                print(f"round={round_id} mode={mode} warmup={not measured} generation={capture['generation_s']:.6f}s graphs={replay.count} spatial={row['summary']['spatial_tokens']}", flush=True)
                del result, prepared
                save()
            errors.append(dict(round=round_id, **{k: metrics(outputs["dense"][k], outputs["sparse"][k]) for k in ("action", "vision")}))
        assert fingerprint(batch) == batch_hash
        stats = {}
        for mode in ("dense", "sparse"):
            stats[mode] = {}
            for scope in ("generation_s", "runner_s"):
                values = [r[scope] for r in rows if r["measured"] and r["mode"] == mode]
                stats[mode][scope] = dict(mean=float(np.mean(values)), median=float(np.median(values)), p90=float(np.percentile(values, 90)))
        placement = {"und_cuda_bytes": 0, "gen_cuda_bytes": 0}
        for layer in model.net.language_model.model.layers:
            for name, tensor in layer.named_parameters():
                if tensor.is_cuda:
                    placement["gen_cuda_bytes" if "moe_gen" in name else "und_cuda_bytes"] += tensor.numel() * tensor.element_size()
        assert placement["und_cuda_bytes"] == 0
        report.update(status="passed", stats=stats, placement=placement, prefill_calls=prefills[0],
                      speedup=stats["dense"]["generation_s"]["median"] / stats["sparse"]["generation_s"]["median"],
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
        save()
        print(json.dumps(dict(stats=stats, speedup=report["speedup"], errors=errors[-1]), indent=2), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        handle.remove()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
