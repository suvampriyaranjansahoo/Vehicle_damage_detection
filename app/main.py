import logging
import time
import uuid
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_EXTENSIONS,
    CLASS_NAMES_PATH,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    MODEL_URL,
    MODEL_VERSION,
)
from .inference import predict
from .labels import load_class_names
from .model_loader import get_model
from .monitoring import log_prediction, summarize_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("vehicle-damage-api")
app = FastAPI(title="Vehicle Damage Detection API", version=MODEL_VERSION)


def _read_bounded_upload(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_upload(file: UploadFile, payload: bytes) -> Image.Image:
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use JPG, JPEG, PNG, or WEBP.")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        with Image.open(BytesIO(payload)) as img:
            img.verify()
        with Image.open(BytesIO(payload)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_configured": bool(MODEL_PATH.exists() or MODEL_URL),
        "classes": len(load_class_names(CLASS_NAMES_PATH)),
        "model_version": MODEL_VERSION,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.get("/ready")
def ready():
    """Readiness probe that verifies the model can actually be loaded."""
    try:
        model = get_model()
        classes = load_class_names(CLASS_NAMES_PATH)
        output_shape = tuple(model.output_shape)
        if len(output_shape) != 2 or output_shape[-1] != len(classes):
            raise ValueError(f"Model output {output_shape} does not match {len(classes)} classes")
        return {"status": "ready", "model_version": MODEL_VERSION, "output_shape": output_shape}
    except Exception as exc:  # noqa: BLE001 -- readiness must report failure without crashing the server
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc


@app.get("/monitoring/summary")
def monitoring_summary(bucket_seconds: int = 3600):
    try:
        return summarize_predictions(bucket_seconds=bucket_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):  # noqa: B008 -- standard FastAPI DI pattern
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    payload = _read_bounded_upload(file)
    image = _validate_upload(file, payload)
    try:
        classes = load_class_names(CLASS_NAMES_PATH)
        result = predict(image, classes)
        result["model_version"] = MODEL_VERSION
        result["request_id"] = request_id
        result["input"] = {"filename": file.filename, "content_type": file.content_type, "bytes": len(payload), "width": image.width, "height": image.height}
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        log_prediction(result, latency_ms=result["latency_ms"])
        return result
    except FileNotFoundError as exc:
        LOGGER.exception("Model artifact missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        LOGGER.exception("Model contract or inference validation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
