#!/usr/bin/env python3
"""Upload PoliTune datasets to Modal volume for training."""

import subprocess
import sys
from pathlib import Path


def main():
    datasets_dir = Path("data/politune_datasets")
    if not datasets_dir.exists():
        print("Error: data/politune_datasets/ not found. Run 04_build_politune_datasets.py first.")
        sys.exit(1)

    conditions = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
    print(f"Uploading {len(conditions)} datasets: {conditions}")

    cmd = [
        "modal", "volume", "put",
        "preference-collapse-data",
        str(datasets_dir),
        "politune_datasets/",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    print("Upload complete.")


if __name__ == "__main__":
    main()
