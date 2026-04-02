#!/usr/bin/env python3
"""Step 4: Build PoliTune datasets for all experimental conditions.

Loads left/right political preference data from PoliTune and builds
SFT and DPO datasets for 6 conditions (+ baseline uses no additional data).
"""

import argparse
import json
from pathlib import Path

from src.politune_data import build_all_politune_datasets


def main():
    parser = argparse.ArgumentParser(description="Build PoliTune datasets")
    parser.add_argument("--output-dir", default="data/politune_datasets",
                        help="Directory to save datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_all_politune_datasets(seed=args.seed)

    # Save all datasets
    metadata = {}
    for name, ds in datasets.items():
        ds_path = output_dir / name
        ds.save_to_disk(str(ds_path))
        metadata[name] = {
            "train_size": len(ds["train"]),
            "eval_size": len(ds["eval"]),
        }
        print(f"Saved {name} to {ds_path}")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll datasets saved to {output_dir}")
    print(f"Upload to Modal: modal volume put preference-collapse-data {output_dir} politune_datasets/")


if __name__ == "__main__":
    main()
