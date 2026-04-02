#!/usr/bin/env python3
"""Step 5b: Train DPO with QLoRA for each experimental condition."""

import argparse

import yaml

from datasets import DatasetDict
from src.training import train_dpo


DPO_CONDITIONS = ["dpo_right", "dpo_left", "dpo_merged"]


def main():
    parser = argparse.ArgumentParser(description="Train DPO models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--condition",
        choices=DPO_CONDITIONS + ["all"],
        default="all",
        help="Which condition to train (default: all)",
    )
    parser.add_argument("--datasets-dir", default="data/politune_datasets",
                        help="Directory containing built datasets")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    conditions = DPO_CONDITIONS if args.condition == "all" else [args.condition]

    print(f"Training conditions: {conditions}")
    print(f"Base model: {config['training']['base_model']}")

    for condition in conditions:
        ds_path = f"{args.datasets_dir}/{condition}"
        dataset = DatasetDict.load_from_disk(ds_path)
        print(f"\n{condition}: train={len(dataset['train'])}, eval={len(dataset['eval'])}")
        train_dpo(dataset, config, condition)

    print("\nDPO training complete.")


if __name__ == "__main__":
    main()
