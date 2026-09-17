"""Remove duplicate rows from data/data.csv before training.

Two kinds of duplication exist in the raw CSV, both of which cause train/test
leakage if left in place (train_test_split has no way to know two rows point
at the same underlying image):

1. Same `image` path listed more than once (identical rows, different index).
2. Different `image` paths whose file contents are byte-identical (the same
   photo saved twice under two names).

This script drops both, keeping the first occurrence, and writes a clean CSV.
Run this once, before scripts/train.py, and pass --csv data/data_dedup.csv to
every downstream script (train.py, fine_tune.py, benchmark_backbones.py).
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/data.csv")
    p.add_argument("--image-root", default=".")
    p.add_argument("--output", default="data/data_dedup.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    n_start = len(df)

    # 1. exact path duplicates
    same_path = df.duplicated(subset="image", keep="first")
    df = df.loc[~same_path].copy()

    # 2. byte-identical content under different filenames
    root = Path(args.image_root)
    seen_hashes = {}
    keep_mask = []
    for img in df["image"]:
        h = hashlib.md5((root / img).read_bytes()).hexdigest()
        if h in seen_hashes:
            keep_mask.append(False)
        else:
            seen_hashes[h] = img
            keep_mask.append(True)
    df = df.loc[keep_mask].copy()

    df.to_csv(args.output, index=False)
    print(f"{n_start} rows -> {len(df)} rows after dedup "
          f"({same_path.sum()} same-path duplicates, "
          f"{n_start - same_path.sum() - len(df)} byte-identical cross-name duplicates removed).")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
