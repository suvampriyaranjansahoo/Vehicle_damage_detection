import json
import time
from collections import Counter
from pathlib import Path

from .config import PREDICTION_LOG_PATH

LOG_PATH = PREDICTION_LOG_PATH


def log_prediction(result: dict, *, latency_ms: float | None = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "predicted_class": result["predicted_class"],
        "confidence": float(result["confidence"]),
        "review_recommended": bool(result.get("confidence_assessment", {}).get("review_recommended", False)),
    }
    if latency_ms is not None:
        record["latency_ms"] = round(float(latency_ms), 2)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def read_predictions(limit: int = 5000) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
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
                if not isinstance(record, dict):
                    continue
                timestamp = float(record["timestamp"])
                confidence = float(record["confidence"])
                if not (0 <= confidence <= 1):
                    continue
                if not isinstance(record["predicted_class"], str):
                    continue
                record["timestamp"] = timestamp
                record["confidence"] = confidence
                records.append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return records[-limit:]


def summarize_predictions(bucket_seconds: int = 3600, limit: int = 5000) -> dict:
    if bucket_seconds < 60 or bucket_seconds > 7 * 24 * 3600:
        raise ValueError("bucket_seconds must be between 60 seconds and 7 days")
    records = read_predictions(limit=limit)
    if not records:
        return {"total_predictions": 0, "review_recommended": 0, "avg_confidence": None, "avg_latency_ms": None, "buckets": [], "class_counts": {}}

    class_counts = Counter(r["predicted_class"] for r in records)
    review_count = sum(bool(r.get("review_recommended", False)) for r in records)
    latencies = [float(r["latency_ms"]) for r in records if isinstance(r.get("latency_ms"), (int, float))]
    bucketed: dict[int, list[dict]] = {}
    for record in records:
        bucket_key = int(float(record["timestamp"]) // bucket_seconds) * bucket_seconds
        bucketed.setdefault(bucket_key, []).append(record)

    buckets = []
    for bucket_start, bucket_records in sorted(bucketed.items()):
        buckets.append({
            "bucket_start": bucket_start,
            "count": len(bucket_records),
            "avg_confidence": round(sum(r["confidence"] for r in bucket_records) / len(bucket_records), 4),
            "review_rate": round(sum(bool(r.get("review_recommended", False)) for r in bucket_records) / len(bucket_records), 4),
        })

    return {
        "total_predictions": len(records),
        "review_recommended": review_count,
        "review_rate": round(review_count / len(records), 4),
        "avg_confidence": round(sum(r["confidence"] for r in records) / len(records), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "buckets": buckets,
        "class_counts": dict(class_counts),
    }
