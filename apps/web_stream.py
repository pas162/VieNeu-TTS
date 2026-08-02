"""Friendly, lazy-loading web frontend for the VieNeu-TTS model family.

Starting this module only serves the web application. The TTS package is imported
and the model is downloaded/initialized after the user explicitly calls
``POST /api/model/load``.

    uv run vieneu-stream                 # http://127.0.0.1:8001
    uv run vieneu-frontend               # friendly alias
"""

from __future__ import annotations

import gc
import importlib.util
import io
import json
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

MAX_REFERENCE_BYTES = 25 * 1024 * 1024
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 15.0
DEFAULT_MODEL_ID = "v3-turbo-fp32"
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "v3-turbo-fp32": {
        "id": "v3-turbo-fp32",
        "name": "VieNeu-TTS v3 Turbo FP32",
        "variant": "FP32 · CPU/ONNX",
        "description": "Chất lượng tối đa, âm thanh chân thực và tự nhiên nhất.",
        "recommended": True,
        "family": "v3 Turbo",
        "device": "CPU",
        "sample_rate": 48_000,
        "clone_mode": "direct",
        "supports_cloning": True,
        "required_modules": [],
        "install_hint": "uv sync --system-certs",
        "load_kwargs": {"mode": "v3turbo", "backend": "onnx", "precision": "fp32"},
    },
    "v3-turbo-int8": {
        "id": "v3-turbo-int8",
        "name": "VieNeu-TTS v3 Turbo INT8",
        "variant": "INT8 · CPU/ONNX",
        "description": "Nhẹ nhất, nhanh trên CPU và phù hợp cho hầu hết máy.",
        "recommended": False,
        "family": "v3 Turbo",
        "device": "CPU",
        "sample_rate": 48_000,
        "clone_mode": "direct",
        "supports_cloning": True,
        "required_modules": [],
        "install_hint": "uv sync --system-certs",
        "load_kwargs": {"mode": "v3turbo", "backend": "onnx", "precision": "int8"},
    },
    "v3-turbo-gpu": {
        "id": "v3-turbo-gpu",
        "name": "VieNeu-TTS v3 Turbo GPU",
        "variant": "PyTorch · NVIDIA GPU",
        "description": "Bản v3 Turbo 48 kHz tăng tốc CUDA, phù hợp máy có GPU NVIDIA.",
        "recommended": False,
        "family": "v3 Turbo",
        "device": "GPU",
        "sample_rate": 48_000,
        "clone_mode": "direct",
        "supports_cloning": True,
        "required_modules": ["torch", "torchaudio", "transformers", "safetensors"],
        "install_hint": "uv sync --group gpu --system-certs",
        "load_kwargs": {"mode": "v3turbo", "backend": "pytorch", "device": "cuda"},
    },
    "v2-turbo-cpu": {
        "id": "v2-turbo-cpu",
        "name": "VieNeu-TTS v2 Turbo GGUF",
        "variant": "GGUF/ONNX · CPU",
        "description": "Bản v2 Turbo gọn nhẹ, chạy CPU và xuất âm thanh 24 kHz.",
        "recommended": False,
        "family": "v2 Turbo",
        "device": "CPU",
        "sample_rate": 24_000,
        "clone_mode": "embedding",
        "supports_cloning": True,
        "required_modules": ["llama_cpp", "librosa"],
        "install_hint": "uv sync --extra legacy --system-certs",
        "load_kwargs": {"mode": "turbo"},
    },
    "v2-turbo-gpu": {
        "id": "v2-turbo-gpu",
        "name": "VieNeu-TTS v2 Turbo GPU",
        "variant": "PyTorch/ONNX · NVIDIA GPU",
        "description": "v2 Turbo chạy CUDA, cân bằng tốc độ, độ nhẹ và khả năng clone giọng.",
        "recommended": False,
        "family": "v2 Turbo",
        "device": "GPU",
        "sample_rate": 24_000,
        "clone_mode": "embedding",
        "supports_cloning": True,
        "required_modules": ["torch", "transformers", "librosa"],
        "install_hint": "uv sync --group gpu --system-certs",
        "load_kwargs": {"mode": "turbo_gpu", "device": "cuda", "backend": "standard"},
    },
    "v2-gpu": {
        "id": "v2-gpu",
        "name": "VieNeu-TTS v2 GPU",
        "variant": "PyTorch · NVIDIA GPU",
        "description": "Model v2 chất lượng cao, song ngữ Việt–Anh và hỗ trợ podcast.",
        "recommended": False,
        "family": "v2",
        "device": "GPU",
        "sample_rate": 24_000,
        "clone_mode": "transcript",
        "supports_cloning": True,
        "required_modules": ["torch", "transformers", "neucodec"],
        "install_hint": "uv sync --group gpu --system-certs",
        "load_kwargs": {
            "mode": "standard",
            "backbone_repo": "pnnbao-ump/VieNeu-TTS-v2",
            "backbone_device": "cuda",
            "codec_repo": "neuphonic/distill-neucodec",
            "codec_device": "cuda",
            "gguf_filename": None,
        },
    },
    "v1-gpu": {
        "id": "v1-gpu",
        "name": "VieNeu-TTS v1 GPU",
        "variant": "PyTorch · NVIDIA GPU",
        "description": "Dòng v1 ổn định dành cho tiếng Việt, preset/clone giọng và âm thanh 24 kHz.",
        "recommended": False,
        "family": "v1",
        "device": "GPU",
        "sample_rate": 24_000,
        "clone_mode": "transcript",
        "supports_cloning": True,
        "required_modules": ["torch", "transformers", "neucodec"],
        "install_hint": "uv sync --group gpu --system-certs",
        "load_kwargs": {
            "mode": "standard",
            "backbone_repo": "pnnbao-ump/VieNeu-TTS",
            "backbone_device": "cuda",
            "codec_repo": "neuphonic/distill-neucodec",
            "codec_device": "cuda",
            "gguf_filename": None,
        },
    },
}
ROOT_DIR = Path(__file__).resolve().parents[1]
CLIENT_HTML_PATH = ROOT_DIR / "client" / "client.html"
V3_VOICES_PATH = ROOT_DIR / "src" / "vieneu" / "assets" / "voices_v3_turbo.json"
LEGACY_VOICES_PATH = ROOT_DIR / "src" / "vieneu" / "assets" / "voices.json"
SAVED_VOICES_DIR = ROOT_DIR / "saved_voices"
SAVED_VOICES_JSON = SAVED_VOICES_DIR / "custom_voices.json"


def _get_custom_voices() -> list[dict[str, Any]]:
    if not SAVED_VOICES_JSON.exists():
        return []
    try:
        return json.loads(SAVED_VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_custom_voice_meta(voice_data: dict[str, Any]) -> None:
    SAVED_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    current = _get_custom_voices()
    current = [v for v in current if v.get("id") != voice_data.get("id")]
    current.insert(0, voice_data)
    SAVED_VOICES_JSON.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _delete_custom_voice_meta(voice_id: str) -> bool:
    current = _get_custom_voices()
    target = next((v for v in current if v.get("id") == voice_id), None)
    if not target:
        return False
    saved_path_str = target.get("saved_path")
    if saved_path_str:
        Path(saved_path_str).unlink(missing_ok=True)
    updated = [v for v in current if v.get("id") != voice_id]
    SAVED_VOICES_JSON.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


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
_selected_model_id = DEFAULT_MODEL_ID
_model_lock = threading.RLock()


def _status_payload() -> dict[str, Any]:
    """Return a serializable snapshot without importing the TTS runtime."""
    with _model_lock:
        config = MODEL_CONFIGS[_selected_model_id]
        return {
            "state": _model_state,
            "ready": _model_state == "ready" and _model is not None,
            "model_id": _selected_model_id,
            "model": config["name"],
            "variant": config["variant"],
            "description": config["description"],
            "sample_rate": config["sample_rate"],
            "supports_cloning": config["supports_cloning"],
            "clone_mode": config["clone_mode"],
            "error": _model_error,
            "loaded_at": _model_loaded_at,
        }


def _missing_requirements(config: dict[str, Any]) -> list[str]:
    """Check optional packages without importing heavyweight runtimes."""
    return [
        module
        for module in config.get("required_modules", [])
        if importlib.util.find_spec(module) is None
    ]


def _public_model_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = _missing_requirements(config)
    public = {
        key: value
        for key, value in config.items()
        if key not in {"load_kwargs", "required_modules"}
    }
    public["available"] = not missing
    public["missing_dependencies"] = missing
    return public


def _load_model(model_id: str) -> None:
    """Import and initialize VieNeu only after an explicit user action."""
    global _model, _model_state, _model_error, _model_loaded_at

    try:
        config = MODEL_CONFIGS[model_id]
        print(f"[VieNeu] Dang tai {config['name']}...")
        from vieneu import Vieneu  # Deliberately lazy: may download model files.

        if config["device"] == "GPU":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Model này cần NVIDIA GPU/CUDA nhưng CUDA chưa sẵn sàng trên máy. "
                    "Hãy chọn bản CPU hoặc kiểm tra driver và bản PyTorch CUDA."
                )

        engine = Vieneu(**config["load_kwargs"])
        with _model_lock:
            _model = engine
            _model_state = "ready"
            _model_error = None
            _model_loaded_at = time.time()
        threads = getattr(getattr(engine, "engine", None), "ort_intra_op_threads", "?")
        print(f"[VieNeu] Model san sang. {config['variant']} | ONNX threads: {threads}")
    except Exception as exc:  # noqa: BLE001 - surface runtime/download errors in UI
        with _model_lock:
            _model = None
            _model_state = "error"
            _model_error = str(exc)
            _model_loaded_at = None
        print(f"[VieNeu] Khong the tai model: {exc}")


def _start_model_load(background_tasks: BackgroundTasks, model_id: str) -> dict[str, Any]:
    global _model, _model_state, _model_error, _model_loaded_at, _selected_model_id

    if model_id not in MODEL_CONFIGS:
        raise HTTPException(status_code=422, detail="Model không được hỗ trợ.")
    config = MODEL_CONFIGS[model_id]
    missing = _missing_requirements(config)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{config['name']} cần cài thêm: {', '.join(missing)}. "
                f"Chạy: {config['install_hint']}"
            ),
        )
    with _model_lock:
        if _model_state == "ready" and _model is not None and _selected_model_id == model_id:
            return _status_payload()
        if _model_state == "loading":
            return _status_payload()
        _model = None
        _model_state = "loading"
        _model_error = None
        _model_loaded_at = None
        _selected_model_id = model_id
        background_tasks.add_task(_load_model, model_id)
    return _status_payload()


def _read_local_voices(model_id: str = DEFAULT_MODEL_ID) -> list[dict[str, str]]:
    """Read bundled voice metadata; this never initializes or downloads a model."""
    config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS[DEFAULT_MODEL_ID])
    if config["family"] != "v3 Turbo":
        with _model_lock:
            engine = _model if _model_state == "ready" and _selected_model_id == model_id else None
        if engine is not None:
            try:
                return [
                    {"id": voice_id, "name": voice_id, "description": label}
                    for label, voice_id in engine.list_preset_voices()
                ]
            except Exception:  # noqa: BLE001 - fall back to the engine default
                pass
    if config["family"] == "v2 Turbo":
        return [{
            "id": "",
            "name": "Giọng mặc định của model",
            "description": "Preset được đọc từ model sau khi tải.",
        }]

    voices_path = V3_VOICES_PATH if config["family"] == "v3 Turbo" else LEGACY_VOICES_PATH
    preset_list = []
    try:
        data = json.loads(voices_path.read_text(encoding="utf-8"))
        presets = data.get("presets", {})
        preset_list = [
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
        preset_list = []

    custom_list = [
        {
            "id": item["id"],
            "name": item["name"],
            "description": item.get("description", "Giọng clone cá nhân"),
            "gender": "Cá nhân",
            "region": "Custom",
            "style": "Clone",
        }
        for item in _get_custom_voices()
    ]
    return custom_list + preset_list


def _ready_model() -> Any:
    global _model, _model_state, _model_error, _model_loaded_at
    with _model_lock:
        if _model_state != "ready" or _model is None:
            _load_model(_selected_model_id)
        if _model_state != "ready" or _model is None:
            raise HTTPException(
                status_code=409,
                detail=f"Khởi động model thất bại: {_model_error or 'Không rõ nguyên nhân'}",
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


@app.get("/api/models")
async def api_models() -> list[dict[str, Any]]:
    """Return the local catalog; listing models never initializes them."""
    return [_public_model_config(config) for config in MODEL_CONFIGS.values()]


class ModelLoadRequest(BaseModel):
    model_id: str = DEFAULT_MODEL_ID


@app.post("/api/model/load", status_code=202)
async def model_load(
    background_tasks: BackgroundTasks,
    request: ModelLoadRequest | None = None,
) -> dict[str, Any]:
    return _start_model_load(background_tasks, (request or ModelLoadRequest()).model_id)


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
async def api_voices(model_id: str = DEFAULT_MODEL_ID) -> list[dict[str, str]]:
    if model_id not in MODEL_CONFIGS:
        model_id = DEFAULT_MODEL_ID
    return _read_local_voices(model_id)


# Backward-compatible route used by the previous streaming client.
@app.get("/voices")
async def voices() -> list[dict[str, str]]:
    return _read_local_voices(DEFAULT_MODEL_ID)


def _pcm16(audio_f32: np.ndarray) -> bytes:
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Nội dung không được để trống.")
    return cleaned


def _validate_reference(path: Path) -> float:
    """Validate a reference clip without importing or initializing the model."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = float(info.duration)
    except Exception:
        try:
            import re
            import subprocess
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            proc = subprocess.run(
                [ffmpeg_exe, "-i", str(path), "-hide_banner"],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                errors="replace",
            )
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:[\.,]\d+)?)", proc.stderr)
            if match:
                h, m, s = match.groups()
                s = s.replace(",", ".")
                duration = float(h) * 3600 + float(m) * 60 + float(s)
            else:
                print(f"[VieNeu] Cannot parse audio duration. FFmpeg output:\n{proc.stderr}")
                raise ValueError("Could not parse audio duration")
        except Exception as exc:  # noqa: BLE001
            print(f"[VieNeu] Reference validation error: {exc}")
            raise HTTPException(
                status_code=422,
                detail=f"Không đọc được audio mẫu ({exc}). Hãy dùng WAV, FLAC, OGG, MP3 hoặc M4A hợp lệ.",
            ) from exc

    if duration < MIN_REFERENCE_SECONDS:
        raise HTTPException(status_code=422, detail="Audio mẫu cần dài ít nhất 1 giây.")
    if duration > MAX_REFERENCE_SECONDS:
        raise HTTPException(status_code=422, detail="Audio mẫu không được dài quá 15 giây.")
    return duration


async def _store_reference(upload: UploadFile) -> tuple[Path, float]:
    """Store a bounded upload in a temporary file and return its duration."""
    suffix_by_type = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/aac": ".m4a",
    }
    original_suffix = Path(upload.filename or "").suffix.lower()
    allowed_suffixes = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".mp4"}
    suffix = suffix_by_type.get(upload.content_type or "")
    if suffix is None and original_suffix in allowed_suffixes:
        suffix = original_suffix
    if suffix is None:
        raise HTTPException(status_code=415, detail="Định dạng hỗ trợ: WAV, FLAC, OGG, MP3 hoặc M4A.")

    temp_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="vieneu-ref-", suffix=suffix, delete=False) as temp:
            temp_path = Path(temp.name)
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_REFERENCE_BYTES:
                    raise HTTPException(status_code=413, detail="Audio mẫu không được vượt quá 25 MB.")
                temp.write(chunk)
        duration = _validate_reference(temp_path)
        return temp_path, duration
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.post("/api/voices/save")
async def save_voice(
    voice_name: str = Form(min_length=1, max_length=100),
    reference_audio: UploadFile = File(...),
) -> dict[str, Any]:
    """Save a reference audio file permanently as a custom voice preset."""
    temp_path, duration = await _store_reference(reference_audio)
    try:
        SAVED_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        safe_filename = f"voice_{int(time.time())}_{temp_path.name}"
        saved_path = SAVED_VOICES_DIR / safe_filename
        import shutil

        shutil.copy(temp_path, saved_path)
        voice_id = f"custom:{saved_path.name}"
        meta = {
            "id": voice_id,
            "name": f"⭐ {voice_name.strip()}",
            "description": f"Giọng clone cá nhân ({duration:.1f}s)",
            "saved_path": str(saved_path),
        }
        _save_custom_voice_meta(meta)
        return {"status": "success", "voice": meta}
    finally:
        temp_path.unlink(missing_ok=True)


@app.delete("/api/voices/custom")
async def delete_custom_voice(voice_id: str = Query(...)) -> dict[str, Any]:
    """Delete a saved custom voice preset and its audio file."""
    success = _delete_custom_voice_meta(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail="Không tìm thấy giọng cần xóa.")
    return {"status": "success", "deleted_id": voice_id}


def _audio_stream(
    text: str,
    voice_id: Optional[str] = None,
    reference_path: Path | None = None,
    reference_text: str | None = None,
) -> StreamingResponse:
    engine = _ready_model()
    config = MODEL_CONFIGS[_selected_model_id]
    sample_rate = int(getattr(engine, "sample_rate", config["sample_rate"]))

    # Resolve custom saved voice if selected in voice_id dropdown
    if reference_path is None and voice_id and voice_id.startswith("custom:"):
        customs = _get_custom_voices()
        matched = next((c for c in customs if c.get("id") == voice_id), None)
        if matched and Path(matched.get("saved_path", "")).exists():
            reference_path = Path(matched["saved_path"])
        else:
            voice_filename = voice_id.replace("custom:", "")
            candidate = SAVED_VOICES_DIR / voice_filename
            if candidate.exists():
                reference_path = candidate

    def generate():
        try:
            header = io.BytesIO()
            with wave.open(header, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.setnframes(1_000_000_000)
            yield header.getvalue()

            started_at = time.perf_counter()
            first_audio_at = None
            emitted = 0
            chunk_count = 0
            if reference_path is not None:
                clone_mode = config["clone_mode"]
                if clone_mode == "direct":
                    infer_kwargs = {"ref_audio": str(reference_path), "denoise": True}
                elif clone_mode == "transcript":
                    infer_kwargs = {
                        "ref_audio": str(reference_path),
                        "ref_text": reference_text,
                    }
                elif clone_mode == "embedding":
                    infer_kwargs = {"ref_codes": engine.encode_reference(str(reference_path))}
                else:
                    raise ValueError(f"{config['name']} không hỗ trợ clone giọng.")
            elif config["family"] == "v3 Turbo":
                clean_voice = voice_id if voice_id and not voice_id.startswith("custom:") else None
                infer_kwargs = {"voice": clean_voice}
            elif voice_id and not voice_id.startswith("custom:"):
                infer_kwargs = {"voice": engine.get_preset_voice(voice_id)}
            else:
                infer_kwargs = {}
            for chunk in engine.infer_stream(text, **infer_kwargs):
                if chunk is None or len(chunk) == 0:
                    continue
                if first_audio_at is None:
                    first_audio_at = time.perf_counter() - started_at
                    print(f"[VieNeu] Time to first audio: {first_audio_at * 1000:.0f} ms")
                emitted += len(chunk)
                chunk_count += 1
                yield _pcm16(chunk)

            duration = emitted / sample_rate
            elapsed = time.perf_counter() - started_at
            rtf = elapsed / duration if duration else 0
            print(f"[VieNeu] {chunk_count} chunks | audio {duration:.2f}s | RTF {rtf:.3f}")
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Audio-Sample-Rate": str(sample_rate),
        },
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


@app.post("/stream/clone")
async def stream_clone(
    text: str = Form(min_length=1, max_length=5_000),
    reference_audio: UploadFile = File(...),
    reference_text: str | None = Form(default=None, max_length=2_000),
) -> StreamingResponse:
    """Clone the uploaded voice and stream synthesized audio; never persist samples."""
    _ready_model()
    config = MODEL_CONFIGS[_selected_model_id]
    if not config["supports_cloning"]:
        await reference_audio.close()
        raise HTTPException(status_code=422, detail=f"{config['name']} không hỗ trợ clone giọng.")
    cleaned_reference_text = (reference_text or "").strip()
    if config["clone_mode"] == "transcript" and not cleaned_reference_text:
        await reference_audio.close()
        raise HTTPException(
            status_code=422,
            detail="Model v1/v2 cần nội dung chính xác của audio mẫu để clone giọng.",
        )
    reference_path, _ = await _store_reference(reference_audio)
    try:
        return _audio_stream(
            _clean_text(text),
            reference_path=reference_path,
            reference_text=cleaned_reference_text or None,
        )
    except Exception:
        reference_path.unlink(missing_ok=True)
        raise


def main() -> None:
    host = "127.0.0.1"
    port = 8001
    print(f"[VieNeu] Studio dang chay tai http://{host}:{port}")
    print("[VieNeu] Model chua duoc tai. Mo giao dien va bam 'Khoi dong model' khi can.")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
