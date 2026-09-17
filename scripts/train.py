"""Reproducible CSV-based training pipeline. Requires the dataset images referenced by data/data.csv."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.labels import save_class_mapping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/data.csv")
    p.add_argument("--image-root", default=".")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="artifacts")
    p.add_argument("--oversample-class", default=None, help="Class name to duplicate in the training split (e.g. a weak/small class)")
    p.add_argument("--oversample-factor", type=int, default=2, help="How many total copies of --oversample-class rows to include (2 = one extra copy)")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not {"image", "classes"}.issubset(df.columns):
        raise ValueError("CSV must contain image and classes columns")
    root = Path(args.image_root)
    df["path"] = df["image"].map(lambda x: str(root / x))
    missing = df.loc[~df["path"].map(lambda x: Path(x).is_file()), "path"]
    if len(missing):
        sample = "\n".join(missing.head(10).tolist())
        raise FileNotFoundError(f"{len(missing)} referenced images are missing. Example paths:\n{sample}")

    # Stratified 70/15/15 split. The test split is written to disk so evaluate.py
    # scores only on rows the model never saw during training or validation --
    # evaluating against the full CSV (train rows included) would leak data and
    # inflate every metric.
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["classes"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["classes"], random_state=SEED)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # Optional oversampling: duplicate a weak/small class's rows within the
    # training split only (never val/test, which must stay a true single copy
    # of unseen data). Combined with the augmentation generator below, each
    # duplicate gets independently randomized transforms per epoch -- so this
    # gives the model more *effective* visual variety on that class, not just
    # literal copies of the same augmented image.
    oversample_note = None
    if args.oversample_class:
        target_rows = train_df[train_df["classes"] == args.oversample_class]
        if len(target_rows) == 0:
            raise ValueError(f"--oversample-class {args.oversample_class!r} matches no rows in the training split")
        extra_copies = pd.concat([target_rows] * (args.oversample_factor - 1), ignore_index=True)
        before = len(train_df)
        train_df = pd.concat([train_df, extra_copies], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        oversample_note = {
            "class": args.oversample_class,
            "factor": args.oversample_factor,
            "original_count": len(target_rows),
            "effective_count": len(target_rows) * args.oversample_factor,
            "train_rows_before": before,
            "train_rows_after": len(train_df),
        }
        print(f"Oversampled {args.oversample_class!r}: {len(target_rows)} -> {len(target_rows) * args.oversample_factor} effective rows")

    aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        horizontal_flip=True,
    )
    plain = ImageDataGenerator(rescale=1.0 / 255)
    common = {
        "x_col": "path",
        "y_col": "classes",
        "target_size": (224, 224),
        "batch_size": args.batch_size,
        "class_mode": "categorical",
    }
    train_gen = aug.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
    val_gen = plain.flow_from_dataframe(val_df, shuffle=False, **common)
    # test_gen intentionally not created here -- scripts/evaluate.py builds its
    # own generator from artifacts/splits/test.csv so evaluation stays a fully
    # separate step from training.

    class_names = save_class_mapping(train_gen.class_indices, Path("class_names.json"))
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=train_gen.classes)
    class_weight = dict(enumerate(weights.tolist()))

    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, class_weight=class_weight)

    # Native .keras format -- NOT .h5. Keras 3's legacy h5 reloader mishandles a
    # Sequential model that wraps a Functional submodel (exactly this architecture)
    # and produces a "expects 1 input(s), but it received 2" error on reload. This
    # bit the model that originally shipped with this repo; saving as .keras avoids
    # it entirely. See scripts/repair_legacy_model.py for how the original file was
    # recovered.
    model.save(out / "mobile_netv2_frozen.keras")

    with (out / "training_config.json").open("w") as f:
        json.dump(
            {
                "class_weight": class_weight,
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
                "oversample": oversample_note,
                "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
            },
            f,
            indent=2,
        )
    print(f"Saved model, class mapping, and held-out test split ({len(test_df)} rows) for evaluation.")


if __name__ == "__main__":
    main()
