import json
import time
from collections import Counter
from pathlib import Path

from .config import PREDICTION_LOG_PATH

LOG_PATH = PREDICTION_LOG_PATH


def log_prediction(result: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "predicted_class": result["predicted_class"],
        "confidence": float(result["confidence"]),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_predictions(limit: int = 5000) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "timestamp" in record and "predicted_class" in record and "confidence" in record:
                    records.append(record)
            except json.JSONDecodeError:
                continue
    return records[-limit:]


def summarize_predictions(bucket_seconds: int = 3600, limit: int = 5000) -> dict:
    if bucket_seconds < 60 or bucket_seconds > 7 * 24 * 3600:
        raise ValueError("bucket_seconds must be between 60 seconds and 7 days")

    records = read_predictions(limit=limit)
    if not records:
        return {"total_predictions": 0, "buckets": [], "class_counts": {}}

    class_counts = Counter(r["predicted_class"] for r in records)
    bucketed: dict[int, list[float]] = {}
    for record in records:
        bucket_key = int(float(record["timestamp"]) // bucket_seconds) * bucket_seconds
        bucketed.setdefault(bucket_key, []).append(float(record["confidence"]))

    buckets = [
        {
            "bucket_start": bucket_start,
            "count": len(confidences),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
        }
        for bucket_start, confidences in sorted(bucketed.items())
    ]

    return {
        "total_predictions": len(records),
        "buckets": buckets,
        "class_counts": dict(class_counts),
    }
