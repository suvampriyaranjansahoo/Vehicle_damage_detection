"""Clean data/data.csv before training: detect near-duplicate images via
perceptual hashing, and optionally drop the `unknown` class.

Why this exists: a public project built on the same source dataset
(hamzamanssor/car-damage-assessment on Kaggle -- the same 1,594-row dataset
this repo's data.csv comes from) found 172 duplicate records in it and chose
to drop the `unknown` class entirely, since it isn't a real damage category
and made up ~34% of the data. This script reproduces that cleaning so you can
compare "trained on raw data.csv" vs "trained on cleaned data" numbers rather
than just taking someone else's word for which is better.

Usage:
    python scripts/preprocess_dataset.py                      # dedup only
    python scripts/preprocess_dataset.py --drop-unknown        # dedup + drop unknown
    python scripts/preprocess_dataset.py --hash-threshold 6    # looser near-duplicate match
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:
    raise SystemExit("Missing dependency: pip install imagehash") from None


def compute_hashes(df: pd.DataFrame) -> list:
    hashes = []
    for path in df["path"]:
        try:
            with Image.open(path) as img:
                hashes.append(imagehash.phash(img))
        except Exception as exc:  # noqa: BLE001 -- report and continue; caller decides what to do with bad rows
            print(f"WARNING: could not hash {path}: {exc}")
            hashes.append(None)
    return hashes


def find_duplicate_indices(hashes: list, threshold: int) -> set:
    """O(n^2) pairwise Hamming-distance comparison. Fine for a few thousand
    images; for a much larger dataset, bucket by exact hash first."""
    duplicates = set()
    n = len(hashes)
    for i in range(n):
        if hashes[i] is None or i in duplicates:
            continue
        for j in range(i + 1, n):
            if hashes[j] is None or j in duplicates:
                continue
            if hashes[i] - hashes[j] <= threshold:
                duplicates.add(j)  # keep the first occurrence, drop later ones
    return duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/data.csv")
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", default="data/data_clean.csv")
    p.add_argument("--drop-unknown", action="store_true", help="Remove the unknown class entirely")
    p.add_argument("--hash-threshold", type=int, default=0, help="Max perceptual-hash Hamming distance to count as a duplicate (0 = exact match only)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    root = Path(args.image_root)
    df["path"] = df["image"].map(lambda x: str(root / x))
    missing = df.loc[~df["path"].map(lambda x: Path(x).is_file())]
    if len(missing):
        sample = "\n".join(missing["path"].head(10).tolist())
        raise FileNotFoundError(
            f"{len(missing)} referenced images are missing under {root}. "
            f"Run scripts/download_dataset.py first. Example paths:\n{sample}"
        )

    original_count = len(df)
    print(f"Original dataset: {original_count} rows")

    print("Computing perceptual hashes (this reads every image once)...")
    df["_hash"] = compute_hashes(df)

    print(f"Scanning for duplicates (hash threshold={args.hash_threshold})...")
    dup_indices = find_duplicate_indices(df["_hash"].tolist(), args.hash_threshold)
    df = df.drop(df.index[list(dup_indices)]).reset_index(drop=True)
    print(f"Removed {len(dup_indices)} duplicate records -> {len(df)} remaining")

    if args.drop_unknown:
        before = len(df)
        df = df[df["classes"] != "unknown"].reset_index(drop=True)
        print(f"Dropped unknown class: {before - len(df)} rows removed -> {len(df)} remaining")

    df = df.drop(columns=["_hash", "path"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved cleaned dataset to {out}")
    print(f"Final: {len(df)} images across {df['classes'].nunique()} classes")
    print(df["classes"].value_counts().to_string())
    print(f"\nTrain on this with: python scripts/train.py --csv {out} --image-root {args.image_root}")


if __name__ == "__main__":
    main()
