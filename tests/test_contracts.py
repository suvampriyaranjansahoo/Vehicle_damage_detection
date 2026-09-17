import json
import zipfile
from pathlib import Path

from app.config import MODEL_PATH, ROOT
from app.labels import load_class_names
from app.monitoring import summarize_predictions


def test_served_model_artifact_has_expected_keras_contract():
    assert MODEL_PATH.is_file()
    with zipfile.ZipFile(MODEL_PATH) as archive:
        config = json.loads(archive.read("config.json"))
    layers = config["config"]["layers"]
    output_layer = layers[-1]
    assert output_layer["class_name"] == "Dense"
    assert output_layer["config"]["units"] == 7
    assert output_layer["config"]["activation"] == "softmax"


def test_class_mapping_matches_served_model():
    labels = load_class_names(ROOT / "class_names.json")
    assert len(labels) == 7
    assert labels == [
        "bumper_dent",
        "bumper_scratch",
        "door_dent",
        "door_scratch",
        "glass_shatter",
        "head_lamp",
        "tail_lamp",
    ]


def test_monitoring_empty_state(monkeypatch, tmp_path):
    import app.monitoring as monitoring

    monkeypatch.setattr(monitoring, "LOG_PATH", Path(tmp_path) / "predictions.jsonl")
    assert summarize_predictions() == {
        "total_predictions": 0,
        "review_recommended": 0,
        "avg_confidence": None,
        "avg_latency_ms": None,
        "buckets": [],
        "class_counts": {},
    }
