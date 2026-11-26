"""Offline policy services and optional Cosmos RoboLab/OpenPI serving."""

import importlib
import json
from pathlib import Path

from .runtime import SparseWAMRunner, activate_runtime


def _native_service_module(backbone):
    activate_runtime(backbone)
    return importlib.import_module("cosmos_framework.scripts.action_policy_server_robolab")


def create_service(
    config,
    *,
    checkpoint=None,
    host="127.0.0.1",
    port=8000,
    seed=0,
    output_dir="outputs/server",
    mode="sparse",
    native_config=None,
):
    """Build an offline policy service; the optional OpenPI transport is not required.

    Checkpoint precedence is explicit argument, native JSON, then backbone default.
    Native settings must retain the fixed Sparse-WAM denoising protocol.
    """
    if mode not in ("dense", "sparse"):
        raise ValueError("mode must be dense or sparse")

    overrides = {}
    if native_config is not None:
        overrides = json.loads(Path(native_config).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("native_config must contain a JSON object")
    if config.backbone == "fastwam-joint":
        from .backends.fastwam_joint import create_joint_service

        return create_joint_service(config, checkpoint, overrides, native_config, mode, seed)
    if overrides.get("sampler", "unipc") != "unipc":
        raise ValueError("Sparse-WAM requires sampler='unipc'")
    config.validate_generation({
        key: overrides[key]
        for key in ("guidance", "num_steps", "shift", "guidance_interval", "normalize_cfg")
        if key in overrides
    })
    selected_checkpoint = (
        checkpoint if checkpoint is not None else overrides.get("checkpoint_path", config.checkpoint)
    )
    if not isinstance(selected_checkpoint, str) or not selected_checkpoint.strip():
        raise ValueError("checkpoint must be a nonempty path or model identifier")
    overrides.update(
        checkpoint_path=selected_checkpoint,
        host=host,
        port=port,
        seed=seed,
        guidance=3.0,
        num_steps=4,
        shift=5.0,
        output_dir=output_dir,
    )
    if config.backbone == "edge":
        overrides["format_prompt_as_json"] = True
    native = _native_service_module(config.backbone)
    args = native.RobolabServerArgs.model_validate(overrides)
    for field in (
        "guidance_interval",
        "hidden_state_capture_dir",
        "block_residual_capture_dir",
        "rope_qk_capture_dir",
    ):
        if getattr(args, field, None) is not None:
            raise ValueError(f"{field} cannot be combined with the Sparse-WAM service")

    class PolicyService(native.RobolabPolicyService):
        def _build_setup_args(self, settings):
            setup = super()._build_setup_args(settings)
            updates = {
                "use_torch_compile": config.backbone == "edge" and config.execution != "eager",
                "use_cuda_graphs": config.backbone == "edge" and config.execution == "compile-graph",
                "dp_replicate_size": 1,
                "dp_shard_size": 1,
                "cp_size": 1,
                "cfgp_size": 1,
            }
            return setup.model_copy(
                update={k: v for k, v in updates.items() if k in type(setup).model_fields}
            )

    service = PolicyService(args)
    service.model.net.pad_for_cuda_graphs = False  # real token lengths, as in the pinned B6 benchmark
    if config.execution != "eager":
        import torch

        # Distinct layers, dense/profile/sparse shapes, and two CFG text lengths.
        key = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        setattr(torch._dynamo.config, key, 256)
        key = (
            "accumulated_recompile_limit"
            if hasattr(torch._dynamo.config, "accumulated_recompile_limit")
            else "accumulated_cache_size_limit"
        )
        setattr(torch._dynamo.config, key, 4096)
    service.sparse_wam_runner = SparseWAMRunner(service.model, config, mode=mode)
    service.model.generate_samples_from_batch = service.sparse_wam_runner
    return service, args


def serve(config, **kwargs):
    """Start the OpenPI WebSocket transport after checking it is installed."""
    if config.backbone == "fastwam-joint":
        raise ValueError("Use create_service for offline FastWAM-Joint LIBERO inference")
    native = _native_service_module(config.backbone)
    try:
        websocket_server = native._load_openpi_websocket_policy_server()
    except ModuleNotFoundError as exc:
        # These transport releases are recorded in vendor/edge/uv.lock.
        raise ModuleNotFoundError(
            "Sparse-WAM WebSocket serving requires a compatible OpenPI transport. "
            "From the source checkout, install it into the Edge environment with "
            "`uv pip install --python .venv/bin/python openpi-server==0.1.0 openpi-client==0.1.2`, "
            "then use "
            "`.venv/bin/python -m sparse_wam serve ...`. An existing compatible full OpenPI "
            "installation is also supported. The root edge extra only covers offline inference; "
            "create_service(...).infer(...) does not require this transport. "
            f"Missing dependency: {exc.name or 'OpenPI transport'}."
        ) from exc
    service, args = create_service(config, **kwargs)

    metadata = {
        "method": "Sparse-WAM",
        **config.run_metadata(kwargs.get("mode", "sparse")),
        "checkpoint": args.checkpoint_path,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    websocket_server(
        policy=service, host=args.host, port=int(args.port), metadata=metadata
    ).serve_forever()
