import json
from pathlib import Path

import pytest

import app.monitoring as monitoring


def test_monitoring_ignores_malformed_and_invalid_records(monkeypatch, tmp_path):
    path = Path(tmp_path) / "predictions.jsonl"
    path.write_text(
        json.dumps({"timestamp": 1, "predicted_class": "door_dent", "confidence": 0.8, "latency_ms": 12}) + "\n"
        "not-json\n"
        + json.dumps({"timestamp": 2, "predicted_class": "bad", "confidence": 4}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(monitoring, "LOG_PATH", path)
    result = monitoring.summarize_predictions(bucket_seconds=60)
    assert result["total_predictions"] == 1
    assert result["avg_confidence"] == 0.8
    assert result["avg_latency_ms"] == 12.0


def test_monitoring_limit_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(monitoring, "LOG_PATH", Path(tmp_path) / "missing.jsonl")
    with pytest.raises(ValueError):
        monitoring.read_predictions(limit=0)
