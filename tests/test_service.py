"""Service configuration and transport checks must not require a GPU or weights."""

import json
from types import SimpleNamespace

import pytest
import torch

from sparse_wam import SparseWAMConfig
from sparse_wam import service


@pytest.fixture
def native_service(monkeypatch):
    state = SimpleNamespace(events=[], server=None)

    class Args:
        @staticmethod
        def model_validate(values):
            return SimpleNamespace(**{"sampler": "unipc", **values})

    class Policy:
        def __init__(self, args):
            state.events.append("model")
            self.model = SimpleNamespace(
                net=SimpleNamespace(),
                sampler=type("UniPCSampler", (), {})(),
                config=SimpleNamespace(
                    rectified_flow_inference_config=SimpleNamespace(scheduler_type=args.sampler)
                ),
                input_caption_key="caption",
                generate_samples_from_batch=lambda *args, **kwargs: {
                    "action": [torch.zeros(32, 8)],
                    "vision": [torch.zeros(1, 9, 2, 2)],
                },
            )

        def infer(self, observation):
            return self.model.generate_samples_from_batch({"caption": [observation["prompt"]]})

    class WebsocketServer:
        def __init__(self, **kwargs):
            state.server = kwargs

        def serve_forever(self):
            state.events.append("serve")

    def transport():
        state.events.append("transport")
        return WebsocketServer

    state.module = SimpleNamespace(
        RobolabPolicyService=Policy,
        RobolabServerArgs=Args,
        _load_openpi_websocket_policy_server=transport,
    )

    def load_module(backbone):
        assert backbone in ("edge", "nano")
        state.events.append("module")
        return state.module

    monkeypatch.setattr(service, "_native_service_module", load_module)
    return state


def native_json(tmp_path, value):
    path = tmp_path / "native.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("backbone", ["edge", "nano"])
@pytest.mark.parametrize("explicit,native,expected", [
    ("explicit-model", {"checkpoint_path": "native-model"}, "explicit-model"),
    (None, {"checkpoint_path": "native-model"}, "native-model"),
    (None, {}, None),
])
def test_checkpoint_precedence(backbone, explicit, native, expected, tmp_path, native_service):
    config = SparseWAMConfig(backbone)
    policy, args = service.create_service(
        config, checkpoint=explicit, native_config=native_json(tmp_path, native)
    )
    assert args.checkpoint_path == (expected or config.checkpoint)
    assert policy.model.generate_samples_from_batch is policy.sparse_wam_runner
    assert "transport" not in native_service.events


@pytest.mark.parametrize("native", [[], None, "settings", 1, True])
def test_native_config_must_be_an_object_before_runtime_import(native, tmp_path, native_service):
    with pytest.raises(ValueError, match="JSON object"):
        service.create_service(SparseWAMConfig("edge"), native_config=native_json(tmp_path, native))
    assert native_service.events == []


def test_malformed_json_does_not_load_model(tmp_path, native_service):
    path = tmp_path / "bad.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        service.create_service(SparseWAMConfig("edge"), native_config=path)
    assert native_service.events == []


@pytest.mark.parametrize("native,message", [
    ({"sampler": "edm"}, "sampler"),
    ({"sampler": None}, "sampler"),
    ({"num_steps": 8}, "num_steps"),
    ({"guidance": 1.0}, "guidance"),
    ({"shift": 1.0}, "shift"),
    ({"normalize_cfg": True}, "unnormalized"),
    ({"guidance_interval": [0, 1]}, "guidance_interval"),
    ({"checkpoint_path": ""}, "checkpoint"),
])
def test_incompatible_settings_fail_before_runtime_import(native, message, tmp_path, native_service):
    with pytest.raises(ValueError, match=message):
        service.create_service(SparseWAMConfig("edge"), native_config=native_json(tmp_path, native))
    assert native_service.events == []


def test_invalid_mode_does_not_load_runtime(native_service):
    with pytest.raises(ValueError, match="mode"):
        service.create_service(SparseWAMConfig("edge"), mode="invalid")
    assert native_service.events == []


def test_matching_native_protocol_is_accepted(tmp_path, native_service):
    settings = {"sampler": "unipc", "guidance": 3.0, "num_steps": 4, "shift": 5.0}
    _, args = service.create_service(
        SparseWAMConfig("edge"), native_config=native_json(tmp_path, settings)
    )
    assert all(getattr(args, key) == value for key, value in settings.items())
    assert "model" in native_service.events


@pytest.mark.parametrize("field", [
    "hidden_state_capture_dir", "block_residual_capture_dir", "rope_qk_capture_dir",
])
def test_capture_settings_are_rejected_before_model_load(field, tmp_path, native_service):
    with pytest.raises(ValueError, match=field):
        service.create_service(
            SparseWAMConfig("edge"), native_config=native_json(tmp_path, {field: "capture"})
        )
    assert "model" not in native_service.events


def test_offline_inference_does_not_load_transport(native_service):
    def unexpected_transport():
        pytest.fail("Offline inference must not require OpenPI")

    native_service.module._load_openpi_websocket_policy_server = unexpected_transport
    policy, _ = service.create_service(SparseWAMConfig("edge"), mode="dense")
    result = policy.infer({"prompt": "pick up the cup"})
    assert tuple(result["action"][0].shape) == (32, 8)
    assert policy.sparse_wam_runner.last_summary["mode"] == "dense"


def test_missing_transport_fails_before_model_load(native_service):
    def missing_transport():
        raise ModuleNotFoundError("missing openpi_server", name="openpi_server")

    native_service.module._load_openpi_websocket_policy_server = missing_transport
    with pytest.raises(ModuleNotFoundError, match="uv pip install --python") as error:
        service.serve(SparseWAMConfig("edge"))
    assert "model" not in native_service.events
    assert "--group" not in str(error.value)
    assert "openpi-server==0.1.0 openpi-client==0.1.2" in str(error.value)
    assert "openpi_server" in str(error.value)


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_serve_checks_transport_first_and_records_effective_checkpoint(
    mode, tmp_path, native_service, capsys,
):
    service.serve(
        SparseWAMConfig("edge"), mode=mode,
        native_config=native_json(tmp_path, {"checkpoint_path": "custom-model"}),
        host="127.0.0.2", port=8123,
    )
    assert native_service.events.index("transport") < native_service.events.index("model")
    assert native_service.events[-1] == "serve"
    server = native_service.server
    assert (server["host"], server["port"]) == ("127.0.0.2", 8123)
    assert server["metadata"]["checkpoint"] == "custom-model"
    assert server["metadata"]["mode"] == mode
    assert json.loads(capsys.readouterr().out) == server["metadata"]
