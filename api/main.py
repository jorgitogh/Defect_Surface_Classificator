from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from ml.src.config import get_paths
from ml.src.infer import DefectPredictor


PREDICTOR: DefectPredictor | None = None


def _looks_like_html(payload: bytes) -> bool:
    head = payload[:512].lstrip().lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or (head.startswith(b"<") and b"<body" in head)
    )


def _download_checkpoint(checkpoint_url: str) -> Path:
    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", Path(tempfile.gettempdir()) / "defect_classifier"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "model_checkpoint.pt"
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            cached_head = f.read(512)
        if not _looks_like_html(cached_head):
            return checkpoint_path
        checkpoint_path.unlink(missing_ok=True)

    request = Request(checkpoint_url, headers={"User-Agent": "defect-classifier-api/1.0"})
    try:
        with urlopen(request, timeout=120) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to download checkpoint: HTTP {response.status}")
            payload = response.read()
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "text/html" in content_type or _looks_like_html(payload):
                raise RuntimeError(
                    "MODEL_CHECKPOINT_URL returned HTML instead of a .pt file. "
                    "Use a direct file URL (for Hugging Face: .../resolve/main/best_model.pt)."
                )
            checkpoint_path.write_bytes(payload)
    except URLError as exc:
        raise RuntimeError(f"Failed to download MODEL_CHECKPOINT_URL: {exc}") from exc

    return checkpoint_path


def _resolve_checkpoint_path(paths) -> Path:
    from_env = os.getenv("MODEL_CHECKPOINT")
    if from_env:
        checkpoint = Path(from_env).expanduser().resolve()
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"MODEL_CHECKPOINT points to a missing file: {checkpoint}")

    run_dirs = sorted(
        [p for p in paths.runs_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        candidate = run_dir / "best_model.pt"
        if candidate.exists():
            return candidate

    checkpoint_url = os.getenv("MODEL_CHECKPOINT_URL")
    if checkpoint_url:
        return _download_checkpoint(checkpoint_url)

    raise FileNotFoundError(
        "No checkpoint found. Set MODEL_CHECKPOINT, MODEL_CHECKPOINT_URL, or ensure ml/runs/run_*/best_model.pt exists."
    )


def _resolve_class_map_path(paths) -> Path:
    from_env = os.getenv("CLASS_MAP_PATH")
    if from_env:
        class_map = Path(from_env).expanduser().resolve()
        if class_map.exists():
            return class_map
        raise FileNotFoundError(f"CLASS_MAP_PATH points to a missing file: {class_map}")

    class_map = paths.processed_metadata_dir / "class_map.json"
    if class_map.exists():
        return class_map
    raise FileNotFoundError(f"class_map.json not found at {class_map}")


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    raw_values = [v.strip() for v in raw.split(",") if v.strip()]
    if not raw_values:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    origins: list[str] = []
    for value in raw_values:
        if value == "*":
            return ["*"]
        if "://" not in value:
            origins.append(f"https://{value}")
        else:
            origins.append(value)

    return origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    global PREDICTOR
    paths = get_paths()
    checkpoint_path = _resolve_checkpoint_path(paths)
    class_map_path = _resolve_class_map_path(paths)
    PREDICTOR = DefectPredictor(checkpoint_path=checkpoint_path, class_map_path=class_map_path)
    print(f"[API] Loaded model from {checkpoint_path}")
    try:
        yield
    finally:
        PREDICTOR = None


app = FastAPI(title="Defect Classifier API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "device": str(PREDICTOR.device),
    }


@app.get("/classes")
def classes() -> dict:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "classes": [
            {"class_id": class_info.class_id, "class_name": class_info.class_name}
            for class_info in PREDICTOR.classes
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Could not decode image file") from exc

    prediction = PREDICTOR.predict(image)
    return {"filename": file.filename, **prediction}
