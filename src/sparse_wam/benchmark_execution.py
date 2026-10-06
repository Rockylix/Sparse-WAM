"""Benchmark-only dispatch: retain native eager Dense while testing compiled Sparse."""

from .config import PaperConfig


class PairedExecution:
    def __init__(self, service, config, *, dense_eager=False):
        self.service, self.config = service, config
        self.dense_eager = dense_eager
        self.variants = None
        if config.backbone == "edge" and config.execution != "eager" and dense_eager:
            net = service.model.net
            heads = ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action")
            compiled_layers = list(net.language_model.model.layers)
            if not all(hasattr(layer, "_orig_mod") for layer in compiled_layers):
                raise RuntimeError("Expected native compiled decoder layers; cannot isolate Dense eager")
            compiled_heads = {name: getattr(net, name) for name in heads}
            if not all(hasattr(fn, "_torchdynamo_orig_callable") for fn in compiled_heads.values()):
                raise RuntimeError("Expected native compiled encode/decode heads")
            self.variants = {
                "dense": (
                    [layer._orig_mod for layer in compiled_layers],
                    {name: fn._torchdynamo_orig_callable for name, fn in compiled_heads.items()},
                ),
                "sparse": (compiled_layers, compiled_heads),
            }

    def select(self, mode):
        runner = self.service.sparse_wam_runner
        runner.mode = mode
        eager = mode == "dense" and (self.dense_eager or self.config.backbone == "nano")
        runner.config = PaperConfig(self.config.backbone, "eager") if eager else self.config
        if self.variants is not None:
            net = self.service.model.net
            layers, heads = self.variants[mode]
            for index, layer in enumerate(layers):
                net.language_model.model.layers[index] = layer
            for name, fn in heads.items():
                setattr(net, name, fn)
            self.service.model.config.compile.enabled = not eager
            self.service.model.config.compile.use_cuda_graphs = (
                not eager and self.config.execution == "compile-graph"
            )


class ReplayCounter:
    """Observe Python CUDA Graph replay calls; this does not request or force capture."""

    def __enter__(self):
        import torch

        self.count = 0
        self.original = torch.cuda.CUDAGraph.replay

        def counted(graph):
            self.count += 1
            return self.original(graph)

        torch.cuda.CUDAGraph.replay = counted
        return self

    def __exit__(self, *exc):
        import torch

        torch.cuda.CUDAGraph.replay = self.original
