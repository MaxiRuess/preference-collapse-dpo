#!/usr/bin/env python3
"""Upload local datasets to Modal volume for training.

Run this once before starting Modal training jobs.

Usage:
    python scripts/modal_upload_data.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    datasets_dir = Path("data/datasets")
    if not datasets_dir.exists():
        print("Error: data/datasets/ not found. Run step 04 first.")
        sys.exit(1)

    # List what we're uploading
    conditions = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
    print(f"Uploading {len(conditions)} datasets: {conditions}")

    # Upload to Modal volume
    cmd = [
        "modal", "volume", "put",
        "preference-collapse-data",
        str(datasets_dir),
        "datasets/",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    print("Upload complete. You can now run: modal run modal_train.py --condition <name>")


if __name__ == "__main__":
    main()
