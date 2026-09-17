import base64
import logging
import math
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .config import CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD, IMAGE_SIZE, MARGIN_THRESHOLD
from .model_loader import get_model, validate_model_contract

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_grad_model(model_id: int, model):
    import tensorflow as tf

    if not model.layers or not isinstance(model.layers[0], tf.keras.Model):
        raise ValueError("Expected a Sequential model with a convolutional backbone at layer 0")
    base = model.layers[0]
    inp = tf.keras.Input(shape=(224, 224, 3))
    conv_features = base(inp, training=False)
    x = conv_features
    for layer in model.layers[1:]:
        x = layer(x, training=False)
    return tf.keras.Model(inp, [conv_features, x])


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, 0)


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / math.log(len(p)))


def _confidence_assessment(probs: np.ndarray) -> dict:
    ordered = np.sort(probs)[::-1]
    top = float(ordered[0])
    second = float(ordered[1]) if len(ordered) > 1 else 0.0
    margin = top - second
    entropy = _entropy(probs)
    # This is a decision-support threshold, not a calibrated probability of correctness.
    low_confidence = top < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD or entropy > ENTROPY_THRESHOLD
    if low_confidence:
        status = "review_recommended"
        note = "Prediction is uncertain; use a clearer image or human review before relying on it."
    else:
        status = "confident"
        note = "Prediction passed the demo confidence checks; it is not a guarantee of correctness."
    return {
        "status": status,
        "review_recommended": low_confidence,
        "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "margin": MARGIN_THRESHOLD, "normalized_entropy": ENTROPY_THRESHOLD},
        "top_probability": round(top, 6),
        "second_probability": round(second, 6),
        "margin": round(margin, 6),
        "normalized_entropy": round(entropy, 6),
        "note": note,
    }


def _build_grad_model(model):
    return _get_grad_model(id(model), model)


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
        "method": "Grad-CAM activation heuristic",
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
        "note": "Illustrative demo estimate, not a repair quote or insurance valuation.",
    }


def predict(image: Image.Image, class_names: list[str]) -> dict:
    model = get_model()
    validate_model_contract(model, len(class_names))
    batch = preprocess(image)
    probs = np.asarray(model.predict(batch, verbose=0))[0]
    if not np.isfinite(probs).all() or probs.ndim != 1 or len(probs) != len(class_names):
        raise ValueError("Model returned an invalid probability vector")
    if np.any(probs < 0) or not np.isclose(float(probs.sum()), 1.0, atol=1e-3):
        raise ValueError("Model returned probabilities that do not form a valid distribution")
    index = int(np.argmax(probs))
    heatmap = gradcam(model, batch, index)
    severity = severity_from_heatmap(heatmap)
    return {
        "predicted_class": class_names[index],
        "confidence": round(float(probs[index]), 6),
        "probabilities": {name: round(float(probs[i]), 6) for i, name in enumerate(class_names)},
        "top_k": [
            {"class": class_names[int(i)], "probability": round(float(probs[int(i)]), 6)}
            for i in np.argsort(probs)[::-1][:3]
        ],
        "confidence_assessment": _confidence_assessment(probs),
        "severity": severity,
        "repair_cost": repair_cost(class_names[index], severity["score"]),
        "gradcam_overlay_base64": overlay_heatmap(image, heatmap),
    }
