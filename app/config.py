import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def _rooted_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


# Native Keras format is the served artifact. The legacy .h5 is retained only
# for provenance and is never selected by default.
MODEL_PATH = _rooted_path(os.getenv("MODEL_PATH", ""), MODEL_DIR / "car_damage_model.keras")
CLASS_NAMES_PATH = _rooted_path(os.getenv("CLASS_NAMES_PATH", ""), ROOT / "class_names.json")
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_CACHE_DIR = _rooted_path(os.getenv("MODEL_CACHE_DIR", ""), ROOT / ".cache")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGE_SIZE = (224, 224)
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
PREDICTION_LOG_PATH = _rooted_path(
    os.getenv("PREDICTION_LOG_PATH", ""), ROOT / "artifacts" / "predictions.jsonl"
)
