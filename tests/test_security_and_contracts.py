from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_predict_rejects_unsupported_extension_before_decoding():
    r = client.post("/predict", files={"file": ("payload.exe", b"not-image", "application/octet-stream")})
    assert r.status_code == 415


def test_predict_accepts_webp_container(monkeypatch):
    fake = {
        "predicted_class": "door_dent",
        "confidence": 0.81,
        "probabilities": {"door_dent": 0.81},
        "confidence_assessment": {"review_recommended": False},
        "severity": {"score": 30, "level": "moderate"},
        "repair_cost": {"currency": "INR", "illustrative": True, "min": 1, "max": 2},
        "gradcam_overlay_base64": "ZmFrZQ==",
    }
    monkeypatch.setattr("app.main.predict", lambda image, classes: fake)
    monkeypatch.setattr("app.main.log_prediction", lambda result, **kwargs: None)
    buf = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="WEBP")
    r = client.post("/predict", files={"file": ("car.webp", buf.getvalue(), "image/webp")})
    assert r.status_code == 200
    body = r.json()
    assert body["input"]["width"] == 64
    assert body["input"]["height"] == 64
    assert "request_id" in body
    assert body["latency_ms"] >= 0
