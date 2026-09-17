import base64
import logging

import cv2
import numpy as np
from PIL import Image

from .config import IMAGE_SIZE
from .model_loader import get_model, validate_model_contract

LOGGER = logging.getLogger(__name__)
_grad_model_cache = {}


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, 0)


def _build_grad_model(model):
    import tensorflow as tf

    key = id(model)
    if key in _grad_model_cache:
        return _grad_model_cache[key]
    if not model.layers or not isinstance(model.layers[1], tf.keras.Model):
        raise ValueError("Expected a Sequential model with a convolutional backbone at layer 1")

    base = model.layers[1]
    inp = tf.keras.Input(shape=(224, 224, 3))
    conv_features = base(inp)
    x = conv_features
    for layer in model.layers[2:]:
        x = layer(x)
    grad_model = tf.keras.Model(inp, [conv_features, x])
    _grad_model_cache[key] = grad_model
    return grad_model


def gradcam(model, batch: np.ndarray, class_index: int) -> np.ndarray:
    import tensorflow as tf

    grad_model = _build_grad_model(model)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        score = predictions[:, class_index]
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise ValueError("Grad-CAM could not compute gradients for the selected class")
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs * pooled[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> str:
    rgb = np.asarray(image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    heat = np.uint8(255 * np.clip(heatmap, 0, 1))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.58, heat, 0.42, 0)
    ok, encoded = cv2.imencode(".jpg", overlay)
    if not ok:
        raise ValueError("Could not encode Grad-CAM overlay")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def severity_from_heatmap(heatmap: np.ndarray) -> dict:
    active = heatmap >= 0.5
    area = float(active.mean())
    intensity = float(heatmap.mean())
    score = round(float(np.clip(0.65 * area + 0.35 * intensity, 0, 1) * 100), 2)
    if score < 25:
        level = "low"
    elif score < 55:
        level = "moderate"
    else:
        level = "high"
    return {
        "score": score,
        "level": level,
        "activation_area": round(area, 4),
        "activation_intensity": round(intensity, 4),
    }


def repair_cost(damage_type: str, severity_score: float) -> dict:
    ranges = {
        "bumper_dent": (2500, 9000),
        "bumper_scratch": (1800, 6500),
        "door_dent": (3000, 12000),
        "door_scratch": (1800, 7000),
        "glass_shatter": (5000, 18000),
        "head_lamp": (4000, 20000),
        "tail_lamp": (3000, 14000),
    }
    low, high = ranges.get(damage_type, (0, 0))
    factor = 0.7 + (float(severity_score) / 100.0) * 0.6
    return {
        "currency": "INR",
        "illustrative": True,
        "min": round(low * factor),
        "max": round(high * factor),
        "note": "Illustrative estimate, not a repair quote.",
    }


def predict(image: Image.Image, class_names: list[str]) -> dict:
    model = get_model()
    validate_model_contract(model, len(class_names))
    batch = preprocess(image)
    probs = np.asarray(model.predict(batch, verbose=0))[0]
    if not np.isfinite(probs).all() or probs.ndim != 1 or len(probs) != len(class_names):
        raise ValueError("Model returned an invalid probability vector")
    index = int(np.argmax(probs))
    heatmap = gradcam(model, batch, index)
    severity = severity_from_heatmap(heatmap)
    return {
        "predicted_class": class_names[index],
        "confidence": round(float(probs[index]), 6),
        "probabilities": {name: round(float(probs[i]), 6) for i, name in enumerate(class_names)},
        "severity": severity,
        "repair_cost": repair_cost(class_names[index], severity["score"]),
        "gradcam_overlay_base64": overlay_heatmap(image, heatmap),
    }
