"""Evaluate a Keras model on a held-out split and save JSON metrics + a confusion matrix PNG.

Defaults to artifacts/splits/test.csv, the split scripts/train.py writes out.
Do NOT point this at the full data.csv -- that file includes the rows the
model was trained on, and evaluating against them would silently inflate
every metric (accuracy, F1, everything) with data the model has already seen.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.config import ROOT
from app.labels import load_class_names


def compute_calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """Bin predictions by their top-class confidence and compare confidence to
    actual accuracy in each bin (reliability diagram data), plus two scalar
    summaries: Expected Calibration Error (ECE) and multiclass Brier score.

    A well-calibrated model's "90% confident" predictions should be right
    about 90% of the time. This function does not tell you whether the model
    is *good*; it tells you whether its confidence numbers can be trusted at
    face value -- relevant here because the API surfaces confidence directly
    to end users alongside the Grad-CAM explanation.
    """
    confidences = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(confidences)
    for lo, hi in itertools.pairwise(bin_edges):
        # last bin is closed on both ends so confidence == 1.0 is included
        in_bin = (confidences > lo) & (confidences <= hi) if hi < 1.0 else (confidences > lo) & (confidences <= hi + 1e-9)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"range": [round(float(lo), 2), round(float(hi), 2)], "count": 0, "avg_confidence": None, "accuracy": None})
            continue
        avg_conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(avg_conf - acc)
        bins.append({
            "range": [round(float(lo), 2), round(float(hi), 2)],
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
        })

    # Multiclass Brier score: mean squared error between predicted probability
    # vector and the one-hot true label, averaged over samples and classes.
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "expected_calibration_error": round(float(ece), 4),
        "brier_score": round(brier, 4),
        "reliability_bins": bins,
        "note": "Lower ECE and Brier score are better. ECE near 0 means confidence "
                "can be read at face value; a model that is e.g. 90% confident on "
                "average within a bin should be correct ~90% of the time in that bin.",
    }


def plot_reliability_diagram(calibration: dict, out_path: Path) -> None:
    bins = [b for b in calibration["reliability_bins"] if b["count"] > 0]
    bin_centers = [(b["range"][0] + b["range"][1]) / 2 for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    confidences = [b["avg_confidence"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, accuracies, width=0.08, alpha=0.7, label="Accuracy", edgecolor="black")
    ax.scatter(bin_centers, confidences, color="red", zorder=5, label="Avg. confidence")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE={calibration['expected_calibration_error']:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--csv",
        default="artifacts/splits/test.csv",
        help="Held-out split to evaluate on. Must NOT be the full training CSV.",
    )
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", default="artifacts/evaluation")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if csv_path.name == "data.csv":
        raise ValueError(
            "Refusing to evaluate against the full data.csv -- it includes training rows. "
            "Run scripts/train.py first, then evaluate against artifacts/splits/test.csv."
        )

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    image_root = Path(args.image_root)
    if not image_root.is_absolute():
        image_root = ROOT / image_root
    df["path"] = df["image"].map(lambda x: str(image_root / x))
    missing = df.loc[~df["path"].map(lambda x: Path(x).is_file())]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} images missing; evaluation cannot run.")

    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        df, x_col="path", y_col="classes", target_size=(224, 224),
        batch_size=32, class_mode="categorical", shuffle=False,
    )
    names = load_class_names(ROOT / "class_names.json")
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = tf.keras.models.load_model(model_path)
    if tuple(model.output_shape)[-1] != len(names):
        raise ValueError(f"Model output shape {model.output_shape} does not match {len(names)} class labels")
    probs = model.predict(gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = gen.classes

    report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    metrics = {
        "evaluated_on": str(csv_path),
        "n_samples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {k: v for k, v in report.items() if k in names},
        "calibration": compute_calibration(probs, y_true),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_reliability_diagram(metrics["calibration"], out / "reliability_diagram.png")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(cm)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Vehicle Damage Confusion Matrix (held-out test, n={len(df)})")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
