import json
import time
from collections import Counter
from pathlib import Path

LOG_PATH = Path("artifacts/predictions.jsonl")


def log_prediction(result: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_predictions(limit: int = 5000) -> list:
    """Read the most recent `limit` logged predictions. Tolerant of a missing
    log file (fresh deployment) and of any single malformed line (in case a
    write is ever interrupted mid-line), since this feeds a dashboard, not a
    correctness-critical path."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


def summarize_predictions(bucket_seconds: int = 3600, limit: int = 5000) -> dict:
    """Aggregate logged predictions into time buckets for a monitoring
    dashboard: per-bucket average confidence and prediction count, plus an
    overall per-class count. Bucketing (default hourly) keeps the payload
    small even after months of traffic, instead of shipping every raw row
    to the client."""
    records = read_predictions(limit=limit)
    if not records:
        return {"total_predictions": 0, "buckets": [], "class_counts": {}}

    class_counts = Counter(r["predicted_class"] for r in records)

    bucketed: dict[int, list] = {}
    for r in records:
        bucket_key = int(r["timestamp"] // bucket_seconds) * bucket_seconds
        bucketed.setdefault(bucket_key, []).append(r["confidence"])

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
