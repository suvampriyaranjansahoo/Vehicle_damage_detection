"""Download the Car Damage Assessment dataset (hamzamanssor/car-damage-assessment)
and lay it out the way data/data.csv expects: images under data/image/.

This project's data/data.csv is the exact same 1,594-row dataset published at
https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment -- confirmed
by row count and class distribution against a public project that documents
using the same source.

Requires a (free) Kaggle account and API token. One-time setup:
  1. https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json

Then:
  pip install kaggle
  python scripts/download_dataset.py

If you'd rather not use the API: download the zip manually from the Kaggle
dataset page above, and unzip it so images end up at data/image/<file>.jpg,
matching the paths already referenced in data/data.csv.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "hamzamanssor/car-damage-assessment"
DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "image"
DOWNLOAD_ZIP = DATA_DIR / "_kaggle_download.zip"


def main():
    if shutil.which("kaggle") is None:
        sys.exit(
            "The `kaggle` CLI isn't installed or isn't on PATH.\n"
            "Run: pip install kaggle\n"
            "Then set up your API token as described in this script's docstring, "
            "and re-run this script. Or download the dataset manually from:\n"
            f"  https://www.kaggle.com/datasets/{DATASET_SLUG}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_SLUG} via the Kaggle API...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(DATA_DIR)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        sys.exit(
            "Kaggle download failed. Common cause: no API token at ~/.kaggle/kaggle.json "
            f"(see this script's docstring for setup).\n\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    zips = list(DATA_DIR.glob("*.zip"))
    if not zips:
        sys.exit("Download reported success but no zip file was found in data/.")
    archive = zips[0]

    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    archive.unlink()

    # The Kaggle archive's internal layout can vary by upload; normalize
    # whatever image folder it contains to data/image/ so it matches data.csv.
    if not IMAGE_DIR.exists():
        candidates = [p for p in DATA_DIR.iterdir() if p.is_dir() and p.name != "image"]
        image_like = [p for p in candidates if any(p.glob("*.jpg")) or any(p.glob("*.jpeg"))]
        if len(image_like) == 1:
            image_like[0].rename(IMAGE_DIR)
        else:
            print(
                f"Could not auto-detect the image folder among {[p.name for p in candidates]}. "
                f"Move/rename whichever one holds the .jpg files to {IMAGE_DIR} manually."
            )
            return

    n_images = len(list(IMAGE_DIR.glob("*.jp*g")))
    print(f"Done. {n_images} images at {IMAGE_DIR}/")
    print("Next: python scripts/preprocess_dataset.py")


if __name__ == "__main__":
    main()
