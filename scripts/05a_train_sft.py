#!/usr/bin/env python3
"""Step 5a: Train SFT Stage 2 — ideology shift on PoliTune data.

Stage 1 (UltraChat) uses pre-built HuggingFaceH4/mistral-7b-sft-beta.
This script handles Stage 2 only: per-condition ideology SFT.
"""

import argparse
import yaml
from datasets import DatasetDict

SFT_CONDITIONS = ["sft_right", "sft_left", "sft_merged"]


def main():
    parser = argparse.ArgumentParser(description="Train SFT Stage 2 (ideology)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", choices=SFT_CONDITIONS + ["all"], default="all")
    parser.add_argument("--datasets-dir", default="data/politune_datasets")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.sft_training import train_sft_ideology
    conditions = SFT_CONDITIONS if args.condition == "all" else [args.condition]

    for condition in conditions:
        ds = DatasetDict.load_from_disk(f"{args.datasets_dir}/{condition}")
        print(f"\n{condition}: train={len(ds['train'])}, eval={len(ds['eval'])}")
        train_sft_ideology(ds, config, condition)


if __name__ == "__main__":
    main()
