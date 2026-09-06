#!/usr/bin/env python3
"""Upload the built PoliTune datasets to the Modal data volume.

Uploads data/politune_datasets/ to politune_datasets_v2/ on the
`preference-collapse-data` volume (the v1 directory on the volume is left
untouched). Run scripts/07_validate_data.py first.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload datasets to Modal")
    parser.add_argument("--datasets-dir", default="data/politune_datasets")
    parser.add_argument("--remote-dir", default="politune_datasets_v2")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not (datasets_dir / "global_split.json").exists():
        print(f"Error: {datasets_dir}/global_split.json not found. Run 04_build_politune_datasets.py first.")
        sys.exit(1)

    conditions = sorted(d.name for d in datasets_dir.iterdir() if d.is_dir())
    print(f"Uploading {len(conditions)} datasets: {conditions} -> {args.remote_dir}/")

    cmd = ["modal", "volume", "put", "--force", "preference-collapse-data",
           str(datasets_dir), args.remote_dir]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    print("Upload complete.")


if __name__ == "__main__":
    main()
