"""Fine-tune the top N layers of the persisted frozen MobileNetV2 model.

The script intentionally reuses artifacts/splits/train.csv and val.csv from the
original training run. This preserves the documented 70/15/15 split and keeps
the held-out test set untouched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.config import ROOT

SEED = 42


def _load_split(path: Path, image_root: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image", "classes"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    df["path"] = df["image"].map(lambda x: str(image_root / x))
    missing = df.loc[~df["path"].map(lambda x: Path(x).is_file()), "path"]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} referenced images are missing; example: {missing.iloc[0]}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="artifacts/splits/train.csv")
    p.add_argument("--val-csv", default="artifacts/splits/val.csv")
    p.add_argument("--image-root", default=".")
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--model", default="model/car_damage_model.keras")
    p.add_argument("--output", default="model/car_damage_model_finetuned.keras")
    args = p.parse_args()

    image_root = Path(args.image_root)
    if not image_root.is_absolute():
        image_root = ROOT / image_root

    train_df = _load_split(ROOT / args.train_csv if not Path(args.train_csv).is_absolute() else Path(args.train_csv), image_root)
    val_df = _load_split(ROOT / args.val_csv if not Path(args.val_csv).is_absolute() else Path(args.val_csv), image_root)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not model_path.is_file():
        raise FileNotFoundError(f"Base model not found: {model_path}")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")

    augment = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    plain = ImageDataGenerator(rescale=1.0 / 255)
    common = {
        "x_col": "path",
        "y_col": "classes",
        "target_size": (224, 224),
        "batch_size": 32,
        "class_mode": "categorical",
    }
    train_gen = augment.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
    val_gen = plain.flow_from_dataframe(val_df, shuffle=False, **common)

    model = tf.keras.models.load_model(model_path)
    if len(model.layers) < 2 or not isinstance(model.layers[1], tf.keras.Model):
        raise ValueError("Expected a Sequential model with a MobileNetV2 Functional backbone at layer 1")
    base = model.layers[1]
    base.trainable = True
    if args.top_n > len(base.layers):
        raise ValueError(f"--top-n={args.top_n} exceeds backbone layer count {len(base.layers)}")
    for layer in base.layers[:-args.top_n]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(train_gen.num_classes),
        y=train_gen.classes,
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=dict(enumerate(weights.tolist())),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    config = {
        "base_model": str(model_path.relative_to(ROOT)),
        "output_model": str(output_path.relative_to(ROOT)),
        "train_split": str(Path(args.train_csv)),
        "validation_split": str(Path(args.val_csv)),
        "test_split": "artifacts/splits/test.csv",
        "top_n": args.top_n,
        "epochs": args.epochs,
        "learning_rate": 1e-5,
        "seed": SEED,
        "class_weight": dict(enumerate(weights.tolist())),
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
    }
    config_path = ROOT / "artifacts" / "fine_tune_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fine-tuned model to {output_path}")


if __name__ == "__main__":
    main()
