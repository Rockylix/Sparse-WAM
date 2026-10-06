from types import SimpleNamespace

import torch

from sparse_wam.benchmark_execution import PairedExecution, ReplayCounter
from sparse_wam.config import PaperConfig


def test_dense_eager_switch_restores_native_layers_and_heads():
    eager = object()
    compiled = SimpleNamespace(_orig_mod=eager)

    def head():
        pass

    def compiled_head():
        pass

    compiled_head._torchdynamo_orig_callable = head
    net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[compiled])))
    heads = ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action")
    for name in heads:
        setattr(net, name, compiled_head)
    service = SimpleNamespace(
        model=SimpleNamespace(net=net, config=SimpleNamespace(compile=SimpleNamespace())),
        sparse_wam_runner=SimpleNamespace(),
    )
    switch = PairedExecution(service, PaperConfig("edge", "compile-graph"), dense_eager=True)
    for _ in range(2):
        switch.select("dense")
        assert net.language_model.model.layers[0] is eager
        assert all(getattr(net, name) is head for name in heads)
        assert service.sparse_wam_runner.config.execution == "eager"
        assert not service.model.config.compile.enabled
        switch.select("sparse")
        assert net.language_model.model.layers[0] is compiled
        assert all(getattr(net, name) is compiled_head for name in heads)
        assert service.sparse_wam_runner.config.execution == "compile-graph"
        assert service.model.config.compile.use_cuda_graphs


def test_replay_counter_restores_method_on_failure(monkeypatch):
    calls = []

    def original(graph):
        calls.append(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", original)
    try:
        with ReplayCounter() as counter:
            torch.cuda.CUDAGraph.replay("sentinel")
            assert counter.count == 1
            raise ValueError()
    except ValueError:
        pass
    assert torch.cuda.CUDAGraph.replay is original
    assert calls == ["sentinel"]
