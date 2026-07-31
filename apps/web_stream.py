"""Friendly, lazy-loading web frontend for VieNeu-TTS v3 Turbo.

Starting this module only serves the web application. The TTS package is imported
and the model is downloaded/initialized after the user explicitly calls
``POST /api/model/load``.

    uv run vieneu-stream                 # http://127.0.0.1:8001
    uv run vieneu-frontend               # friendly alias
"""

from __future__ import annotations

import gc
import io
import json
import threading
import time
import wave
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

SAMPLE_RATE = 48_000
MODEL_NAME = "VieNeu-TTS-v3-Turbo"
MODEL_VARIANT = "int8 · CPU/ONNX"
ROOT_DIR = Path(__file__).resolve().parents[1]
CLIENT_HTML_PATH = ROOT_DIR / "client" / "client.html"
VOICES_PATH = ROOT_DIR / "src" / "vieneu" / "assets" / "voices_v3_turbo.json"

app = FastAPI(
    title="VieNeu Studio",
    description="Lazy-loading frontend for VieNeu-TTS",
    version="1.0.0",
)

ModelState = Literal["idle", "loading", "ready", "error"]
_model: Any | None = None
_model_state: ModelState = "idle"
_model_error: str | None = None
_model_loaded_at: float | None = None
_model_lock = threading.RLock()


def _status_payload() -> dict[str, Any]:
    """Return a serializable snapshot without importing the TTS runtime."""
    with _model_lock:
        return {
            "state": _model_state,
            "ready": _model_state == "ready" and _model is not None,
            "model": MODEL_NAME,
            "variant": MODEL_VARIANT,
            "sample_rate": SAMPLE_RATE,
            "error": _model_error,
            "loaded_at": _model_loaded_at,
        }


def _load_model() -> None:
    """Import and initialize VieNeu only after an explicit user action."""
    global _model, _model_state, _model_error, _model_loaded_at

    try:
        print("[VieNeu] Dang tai VieNeu-TTS v3 Turbo (int8, CPU)...")
        from vieneu import Vieneu  # Deliberately lazy: may download model files.

        engine = Vieneu(backend="onnx", precision="int8")
        with _model_lock:
            _model = engine
            _model_state = "ready"
            _model_error = None
            _model_loaded_at = time.time()
        threads = getattr(getattr(engine, "engine", None), "ort_intra_op_threads", "?")
        print(f"[VieNeu] Model san sang. Backbone: int8 | ONNX threads: {threads}")
    except Exception as exc:  # noqa: BLE001 - surface runtime/download errors in UI
        with _model_lock:
            _model = None
            _model_state = "error"
            _model_error = str(exc)
            _model_loaded_at = None
        print(f"[VieNeu] Khong the tai model: {exc}")


def _start_model_load(background_tasks: BackgroundTasks) -> dict[str, Any]:
    global _model_state, _model_error

    with _model_lock:
        if _model_state == "ready" and _model is not None:
            return _status_payload()
        if _model_state == "loading":
            return _status_payload()
        _model_state = "loading"
        _model_error = None
        background_tasks.add_task(_load_model)
    return _status_payload()


def _read_local_voices() -> list[dict[str, str]]:
    """Read bundled voice metadata; this never initializes or downloads a model."""
    try:
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
        presets = data.get("presets", {})
        return [
            {
                "id": name,
                "name": name,
                "description": details.get("description", "Giọng đọc VieNeu"),
                "gender": details.get("gender", ""),
                "region": details.get("region", ""),
                "style": details.get("style", ""),
            }
            for name, details in presets.items()
        ]
    except (OSError, ValueError, TypeError):
        return []


def _ready_model() -> Any:
    with _model_lock:
        if _model_state != "ready" or _model is None:
            raise HTTPException(
                status_code=409,
                detail="Model chưa sẵn sàng. Hãy bấm ‘Khởi động model’ trước.",
            )
        return _model


@app.get("/")
async def ui() -> Response:
    if CLIENT_HTML_PATH.exists():
        return FileResponse(CLIENT_HTML_PATH, media_type="text/html")
    return Response("client.html not found", status_code=404, media_type="text/plain")


@app.get("/favicon.ico")
async def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect width="100" height="100" rx="24" fill="#171c19"/>'
        '<text x="50" y="69" text-anchor="middle" font-size="54">🎙️</text></svg>'
    )
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/status")
async def model_status() -> dict[str, Any]:
    return _status_payload()


@app.post("/api/model/load", status_code=202)
async def model_load(background_tasks: BackgroundTasks) -> dict[str, Any]:
    return _start_model_load(background_tasks)


@app.post("/api/model/unload")
async def model_unload() -> dict[str, Any]:
    global _model, _model_state, _model_error, _model_loaded_at

    with _model_lock:
        if _model_state == "loading":
            raise HTTPException(status_code=409, detail="Model đang tải, chưa thể giải phóng.")
        _model = None
        _model_state = "idle"
        _model_error = None
        _model_loaded_at = None
    gc.collect()
    return _status_payload()


@app.get("/api/voices")
async def api_voices() -> list[dict[str, str]]:
    return _read_local_voices()


# Backward-compatible route used by the previous streaming client.
@app.get("/voices")
async def voices() -> list[dict[str, str]]:
    return _read_local_voices()


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Nội dung không được để trống.")
    return cleaned


def _audio_stream(text: str, voice_id: Optional[str]) -> StreamingResponse:
    engine = _ready_model()

    def generate():
        header = io.BytesIO()
        with wave.open(header, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.setnframes(1_000_000_000)
        yield header.getvalue()

        started_at = time.perf_counter()
        first_audio_at = None
        emitted = 0
        chunk_count = 0
        for chunk in engine.infer_stream(text, voice=voice_id or None):
            if chunk is None or len(chunk) == 0:
                continue
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - started_at
                print(f"[VieNeu] Time to first audio: {first_audio_at * 1000:.0f} ms")
            emitted += len(chunk)
            chunk_count += 1
            yield _pcm16(chunk)

        duration = emitted / SAMPLE_RATE
        elapsed = time.perf_counter() - started_at
        rtf = elapsed / duration if duration else 0
        print(f"[VieNeu] {chunk_count} chunks | audio {duration:.2f}s | RTF {rtf:.3f}")

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/stream")
async def stream_get(
    text: str = Query(min_length=1, max_length=5_000),
    voice_id: Optional[str] = None,
) -> StreamingResponse:
    return _audio_stream(_clean_text(text), voice_id)


class StreamRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)
    voice_id: Optional[str] = None


@app.post("/stream")
async def stream_post(request: StreamRequest) -> StreamingResponse:
    return _audio_stream(_clean_text(request.text), request.voice_id)


def main() -> None:
    host = "127.0.0.1"
    port = 8001
    print(f"[VieNeu] Studio dang chay tai http://{host}:{port}")
    print("[VieNeu] Model chua duoc tai. Mo giao dien va bam 'Khoi dong model' khi can.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
