import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
# car_damage_model.keras (native format) is the repaired, loadable version of the
# original car_damage_model.h5 -- see scripts/repair_legacy_model.py for why the
# original file can't be loaded directly under Keras 3.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "car_damage_model.keras")))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", str(ROOT / "class_names.json")))
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(ROOT / ".cache")))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGE_SIZE = (224, 224)
