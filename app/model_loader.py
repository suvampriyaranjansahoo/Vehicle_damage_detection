import os
import threading
import urllib.request
from pathlib import Path

from .config import MODEL_CACHE_DIR, MODEL_PATH, MODEL_URL

_model = None
_lock = threading.Lock()


def _resolve_model_path() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    hf_repo = os.getenv("HF_MODEL_ID", "")
    hf_filename = os.getenv("HF_MODEL_FILENAME", "car_damage_model.h5")
    if hf_repo:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=hf_repo, filename=hf_filename, cache_dir=MODEL_CACHE_DIR))
    if not MODEL_URL:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Set MODEL_URL or HF_MODEL_ID.")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_CACHE_DIR / Path(MODEL_URL.split("?")[0]).name
    if not target.exists():
        urllib.request.urlretrieve(MODEL_URL, target)
    return target


def get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            import tensorflow as tf
            _model = tf.keras.models.load_model(_resolve_model_path())
    return _model
