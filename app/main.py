import logging
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import (
    ALLOWED_EXTENSIONS,
    CLASS_NAMES_PATH,
    MAX_UPLOAD_BYTES,
    MODEL_PATH,
    MODEL_URL,
)
from .inference import predict
from .labels import load_class_names
from .monitoring import log_prediction, summarize_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("vehicle-damage-api")
app = FastAPI(title="Vehicle Damage Detection API", version="1.0.0")


def _validate_upload(file: UploadFile, payload: bytes):
    ext = (file.filename or "").lower().rsplit(".", 1)[-1] if "." in (file.filename or "") else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use JPG, JPEG, PNG, or WEBP.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    try:
        img = Image.open(BytesIO(payload))
        img.verify()
        img = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc
    return img


@app.get("/health")
def health():
    return {"status": "ok", "model_configured": bool(MODEL_PATH.exists() or MODEL_URL), "classes": len(load_class_names(CLASS_NAMES_PATH))}


@app.get("/monitoring/summary")
def monitoring_summary(bucket_seconds: int = 3600):
    """Aggregated view of logged predictions for the Streamlit monitoring tab:
    prediction volume and average confidence per time bucket, plus a
    per-class count. Reads app/monitoring.py's JSONL log rather than a
    database, matching the rest of this project's no-extra-infra approach."""
    return summarize_predictions(bucket_seconds=bucket_seconds)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):  # noqa: B008 -- standard FastAPI DI pattern
    payload = await file.read()
    image = _validate_upload(file, payload)
    try:
        classes = load_class_names(CLASS_NAMES_PATH)
        result = predict(image, classes)
        log_prediction(result)
        return result
    except FileNotFoundError as exc:
        LOGGER.exception("Model artifact missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from exc
