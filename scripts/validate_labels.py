"""Validate that class_names.json matches the labels actually present in the
training split for the model currently served (model/car_damage_model.keras).

This intentionally checks against artifacts/splits/train.csv, not the raw
data/data.csv -- the served model was trained on a deduplicated split with
the `unknown` class dropped (see scripts/preprocess_dataset.py), so its label
set is a strict subset of the raw CSV's. Comparing against the raw CSV would
fail here even though the mapping is correct for the model that's live.
"""
import json
from pathlib import Path

import pandas as pd

TRAIN_SPLIT = Path("artifacts/splits/train.csv")
MAPPING = Path("class_names.json")

if not TRAIN_SPLIT.exists():
    raise SystemExit(
        f"{TRAIN_SPLIT} not found -- run scripts/preprocess_dataset.py and scripts/train.py first, "
        "or this script has nothing to validate the mapping against."
    )

df = pd.read_csv(TRAIN_SPLIT)
labels = sorted(df["classes"].dropna().unique().tolist())
mapping = json.loads(MAPPING.read_text())
ordered = [mapping[str(i)] for i in range(len(mapping))]
assert ordered == labels, f"Mapping mismatch. expected sorted train-split labels {labels}, got {ordered}"
print("Label mapping validated:", ordered)
