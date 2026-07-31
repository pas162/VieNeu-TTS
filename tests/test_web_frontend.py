"""Tests for the lazy-loading VieNeu Studio frontend."""

import io
import subprocess
import sys
import types
import wave
from pathlib import Path

import numpy as np


def _wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return output.getvalue()


def test_import_does_not_import_vieneu_or_start_model():
    """A fresh frontend process must not touch the model runtime at startup."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import apps.web_stream as web; "
                "assert 'vieneu' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert web._status_payload()['state'] == 'idle'"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_voice_list_comes_from_bundled_metadata():
    from apps import web_stream

    voices = web_stream._read_local_voices()
    assert len(voices) == 14
    assert all({"id", "name", "description"} <= voice.keys() for voice in voices)
    assert len(web_stream._read_local_voices("v2-gpu")) == 6
    assert web_stream._read_local_voices("v2-turbo-cpu")[0]["id"] == ""


def test_public_frontend_api_works_before_model_load(monkeypatch):
    from fastapi.testclient import TestClient
    from apps import web_stream

    monkeypatch.setattr(web_stream, "_model", None)
    monkeypatch.setattr(web_stream, "_model_state", "idle")
    monkeypatch.setattr(web_stream, "_selected_model_id", web_stream.DEFAULT_MODEL_ID)
    client = TestClient(web_stream.app)

    assert client.get("/").status_code == 200
    status = client.get("/api/status").json()
    assert status["state"] == "idle"
    assert status["ready"] is False
    models = client.get("/api/models").json()
    assert [model["id"] for model in models] == [
        "v3-turbo-int8",
        "v3-turbo-fp32",
        "v3-turbo-gpu",
        "v2-turbo-cpu",
        "v2-turbo-gpu",
        "v2-gpu",
        "v1-gpu",
    ]
    assert models[0]["recommended"] is True
    assert {model["family"] for model in models} == {"v1", "v2", "v2 Turbo", "v3 Turbo"}
    assert all("available" in model for model in models)
    assert client.post("/api/model/load", json={"model_id": "unknown"}).status_code == 422
    assert len(client.get("/api/voices").json()) == 14
    assert client.post("/stream", json={"text": "Xin chào"}).status_code == 409
    assert client.post("/stream", json={"text": "   "}).status_code == 422

    stream_calls = []

    class FakeStreamEngine:
        @staticmethod
        def infer_stream(text, **kwargs):
            stream_calls.append((text, kwargs))
            yield np.array([0.0, 0.25, -0.25], dtype=np.float32)

    monkeypatch.setattr(web_stream, "_model", FakeStreamEngine())
    monkeypatch.setattr(web_stream, "_model_state", "ready")
    audio = client.post("/stream", json={"text": "Xin chào", "voice_id": "Minh Đức"})
    assert audio.status_code == 200
    assert audio.content.startswith(b"RIFF")
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.headers["x-audio-sample-rate"] == "48000"
    assert stream_calls[0] == ("Xin chào", {"voice": "Minh Đức"})

    cloned = client.post(
        "/stream/clone",
        data={"text": "Giọng nói đã clone"},
        files={"reference_audio": ("sample.wav", _wav_bytes(3), "audio/wav")},
    )
    assert cloned.status_code == 200
    assert cloned.content.startswith(b"RIFF")
    clone_text, clone_kwargs = stream_calls[1]
    assert clone_text == "Giọng nói đã clone"
    assert clone_kwargs["denoise"] is True
    reference_path = Path(clone_kwargs["ref_audio"])
    assert reference_path.name.startswith("vieneu-ref-")
    assert not reference_path.exists(), "temporary voice sample must be deleted after streaming"

    too_short = client.post(
        "/stream/clone",
        data={"text": "Không hợp lệ"},
        files={"reference_audio": ("short.wav", _wav_bytes(0.5), "audio/wav")},
    )
    assert too_short.status_code == 422


def test_model_is_loaded_only_when_loader_runs(monkeypatch):
    from apps import web_stream

    calls = []

    class FakeEngine:
        ort_intra_op_threads = 2

    def fake_vieneu(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(engine=FakeEngine())

    class ImmediateTasks:
        @staticmethod
        def add_task(function, *args):
            function(*args)

    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=fake_vieneu))
    monkeypatch.setattr(web_stream, "_model", None)
    monkeypatch.setattr(web_stream, "_model_state", "idle")
    monkeypatch.setattr(web_stream, "_selected_model_id", web_stream.DEFAULT_MODEL_ID)

    int8_status = web_stream._start_model_load(ImmediateTasks(), "v3-turbo-int8")
    assert int8_status["ready"] is True
    fp32_status = web_stream._start_model_load(ImmediateTasks(), "v3-turbo-fp32")

    assert calls == [
        {"mode": "v3turbo", "backend": "onnx", "precision": "int8"},
        {"mode": "v3turbo", "backend": "onnx", "precision": "fp32"},
    ]
    assert fp32_status["model_id"] == "v3-turbo-fp32"
    assert fp32_status["ready"] is True


def test_legacy_model_uses_24khz_voice_objects_and_clone_transcript(monkeypatch):
    from fastapi.testclient import TestClient
    from apps import web_stream

    calls = []

    class FakeLegacyEngine:
        sample_rate = 24_000

        @staticmethod
        def get_preset_voice(voice_id):
            return {"codes": [1, 2], "text": f"preset:{voice_id}"}

        @staticmethod
        def infer_stream(text, **kwargs):
            calls.append((text, kwargs))
            yield np.array([0.1, -0.1], dtype=np.float32)

    monkeypatch.setattr(web_stream, "_model", FakeLegacyEngine())
    monkeypatch.setattr(web_stream, "_model_state", "ready")
    monkeypatch.setattr(web_stream, "_selected_model_id", "v2-gpu")
    client = TestClient(web_stream.app)

    preset = client.post("/stream", json={"text": "Xin chào", "voice_id": "Binh"})
    assert preset.status_code == 200
    assert preset.headers["x-audio-sample-rate"] == "24000"
    with wave.open(io.BytesIO(preset.content), "rb") as audio:
        assert audio.getframerate() == 24_000
    assert calls[0][1]["voice"]["text"] == "preset:Binh"

    missing_text = client.post(
        "/stream/clone",
        data={"text": "Clone"},
        files={"reference_audio": ("sample.wav", _wav_bytes(3), "audio/wav")},
    )
    assert missing_text.status_code == 422

    cloned = client.post(
        "/stream/clone",
        data={"text": "Clone", "reference_text": "Đây là nội dung mẫu."},
        files={"reference_audio": ("sample.wav", _wav_bytes(3), "audio/wav")},
    )
    assert cloned.status_code == 200
    assert calls[1][1]["ref_text"] == "Đây là nội dung mẫu."
    assert not Path(calls[1][1]["ref_audio"]).exists()


def test_turbo_clone_encodes_reference_once(monkeypatch):
    from fastapi.testclient import TestClient
    from apps import web_stream

    calls = []

    class FakeTurboEngine:
        sample_rate = 24_000

        @staticmethod
        def encode_reference(path):
            assert Path(path).exists()
            return np.array([[0.25, 0.5]], dtype=np.float32)

        @staticmethod
        def infer_stream(text, **kwargs):
            calls.append(kwargs)
            yield np.array([0.0], dtype=np.float32)

    monkeypatch.setattr(web_stream, "_model", FakeTurboEngine())
    monkeypatch.setattr(web_stream, "_model_state", "ready")
    monkeypatch.setattr(web_stream, "_selected_model_id", "v2-turbo-cpu")
    client = TestClient(web_stream.app)

    response = client.post(
        "/stream/clone",
        data={"text": "Turbo clone"},
        files={"reference_audio": ("sample.wav", _wav_bytes(3), "audio/wav")},
    )
    assert response.status_code == 200
    np.testing.assert_array_equal(calls[0]["ref_codes"], [[0.25, 0.5]])
