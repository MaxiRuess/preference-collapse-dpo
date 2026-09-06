#!/usr/bin/env python3
"""Step 4: Build PoliTune SFT datasets for all experimental conditions.

Makes ONE global prompt-level split shared by every condition, then builds
sft_right, sft_left and one sft_merged_s{seed} dataset per label-flip seed.
Run scripts/07_validate_data.py afterwards before uploading to Modal.
"""

import argparse
import json
import shutil
from pathlib import Path

from src.politune_data import build_all_politune_datasets, save_global_split


def main():
    parser = argparse.ArgumentParser(description="Build PoliTune datasets")
    parser.add_argument("--output-dir", default="data/politune_datasets")
    parser.add_argument("--seed", type=int, default=42, help="Global split seed")
    parser.add_argument("--flip-seeds", default="42,43,44",
                        help="Comma-separated label-flip seeds for sft_merged")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    flip_seeds = tuple(int(x) for x in args.flip_seeds.split(","))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale condition directories from the v1 pipeline.
    for stale in output_dir.glob("dpo_*"):
        print(f"Removing stale v1 dataset: {stale}")
        shutil.rmtree(stale)
    old_merged = output_dir / "sft_merged"
    if old_merged.exists():
        print(f"Removing stale v1 dataset: {old_merged}")
        shutil.rmtree(old_merged)

    datasets, split = build_all_politune_datasets(
        seed=args.seed, flip_seeds=flip_seeds, eval_ratio=args.eval_ratio,
    )

    save_global_split(split, output_dir / "global_split.json")
    print(f"Saved global split to {output_dir / 'global_split.json'}")

    metadata = {"global_split": split["stats"], "conditions": {}}
    for name, ds in datasets.items():
        ds_path = output_dir / name
        if ds_path.exists():
            shutil.rmtree(ds_path)
        ds.save_to_disk(str(ds_path))
        metadata["conditions"][name] = {
            "train_size": len(ds["train"]),
            "eval_size": len(ds["eval"]),
        }
        print(f"Saved {name} to {ds_path}")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nAll datasets saved to {output_dir}")
    print("Next: PYTHONPATH=. python scripts/07_validate_data.py")


if __name__ == "__main__":
    main()
