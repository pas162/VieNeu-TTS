"""Tests for the lazy-loading VieNeu Studio frontend."""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np


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


def test_public_frontend_api_works_before_model_load(monkeypatch):
    from fastapi.testclient import TestClient
    from apps import web_stream

    monkeypatch.setattr(web_stream, "_model", None)
    monkeypatch.setattr(web_stream, "_model_state", "idle")
    client = TestClient(web_stream.app)

    assert client.get("/").status_code == 200
    status = client.get("/api/status").json()
    assert status["state"] == "idle"
    assert status["ready"] is False
    assert len(client.get("/api/voices").json()) == 14
    assert client.post("/stream", json={"text": "Xin chào"}).status_code == 409
    assert client.post("/stream", json={"text": "   "}).status_code == 422

    class FakeStreamEngine:
        @staticmethod
        def infer_stream(text, voice=None):
            assert text == "Xin chào"
            assert voice == "Minh Đức"
            yield np.array([0.0, 0.25, -0.25], dtype=np.float32)

    monkeypatch.setattr(web_stream, "_model", FakeStreamEngine())
    monkeypatch.setattr(web_stream, "_model_state", "ready")
    audio = client.post("/stream", json={"text": "Xin chào", "voice_id": "Minh Đức"})
    assert audio.status_code == 200
    assert audio.content.startswith(b"RIFF")
    assert audio.headers["content-type"].startswith("audio/wav")


def test_model_is_loaded_only_when_loader_runs(monkeypatch):
    from apps import web_stream

    calls = []

    class FakeEngine:
        ort_intra_op_threads = 2

    def fake_vieneu(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(engine=FakeEngine())

    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=fake_vieneu))
    monkeypatch.setattr(web_stream, "_model", None)
    monkeypatch.setattr(web_stream, "_model_state", "loading")
    web_stream._load_model()

    assert calls == [{"backend": "onnx", "precision": "int8"}]
    assert web_stream._status_payload()["ready"] is True
