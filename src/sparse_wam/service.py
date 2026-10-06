"""Shared RoboLab/openpi service entry; native preprocessing and action semantics remain intact."""

import json
from pathlib import Path

from .runtime import SparseWAMRunner, activate_runtime


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
    activate_runtime(config.backbone)
    from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs

    overrides = {}
    if native_config is not None:
        overrides = json.loads(Path(native_config).read_text(encoding="utf-8"))
    overrides.update(
        checkpoint_path=checkpoint or config.checkpoint,
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
    args = RobolabServerArgs.model_validate(overrides)
    for field in (
        "guidance_interval",
        "hidden_state_capture_dir",
        "block_residual_capture_dir",
        "rope_qk_capture_dir",
    ):
        if getattr(args, field, None) is not None:
            raise ValueError(f"{field} cannot be combined with the Sparse-WAM service")

    class PolicyService(RobolabPolicyService):
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
    service, args = create_service(config, **kwargs)
    from cosmos_framework.scripts.action_policy_server_robolab import _load_openpi_websocket_policy_server

    metadata = {"method": "Sparse-WAM", **config.run_metadata(kwargs.get("mode", "sparse"))}
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    _load_openpi_websocket_policy_server()(
        policy=service, host=args.host, port=int(args.port), metadata=metadata
    ).serve_forever()
