#!/usr/bin/env python3
"""Step 5a: Train ideology SFT runs locally (one per condition x seed).

The Modal equivalent is `modal run modal_train.py --condition all --seeds 42,43,44`.
"""

import argparse

import yaml
from datasets import DatasetDict

SFT_CONDITIONS = ["sft_right", "sft_left", "sft_merged"]


def main():
    parser = argparse.ArgumentParser(description="Train ideology SFT runs")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", choices=SFT_CONDITIONS + ["all"], default="all")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default: training.seeds from config)")
    parser.add_argument("--datasets-dir", default="data/politune_datasets")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.sft_training import train_sft_ideology
    conditions = SFT_CONDITIONS if args.condition == "all" else [args.condition]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(config["training"].get("seeds", [42])))

    for condition in conditions:
        for seed in seeds:
            ds_name = f"sft_merged_s{seed}" if condition == "sft_merged" else condition
            ds = DatasetDict.load_from_disk(f"{args.datasets_dir}/{ds_name}")
            print(f"\n{condition} seed={seed}: train={len(ds['train'])}, eval={len(ds['eval'])}")
            train_sft_ideology(ds, config, condition, seed)


if __name__ == "__main__":
    main()
