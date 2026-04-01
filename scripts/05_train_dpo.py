#!/usr/bin/env python3
"""Step 5: Train DPO with QLoRA for each experimental condition."""

import argparse

import yaml

from src.dataset_builder import load_datasets
from src.training import train_all_conditions, train_dpo, train_multi_adapter

ALL_CONDITIONS = [
    "dpo_optimist",
    "dpo_skeptic",
    "dpo_merged",
    "dpo_multi",
    "dpo_conf_opt_unc_skp",
    "dpo_conf_skp_unc_opt",
]


def main():
    parser = argparse.ArgumentParser(description="Train DPO models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--condition",
        choices=ALL_CONDITIONS + ["all"],
        default="all",
        help="Which condition to train (default: all)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datasets = load_datasets(config["paths"]["datasets_dir"])

    if args.condition != "all":
        datasets = {args.condition: datasets[args.condition]}

    print(f"Training conditions: {list(datasets.keys())}")
    print(f"Base model: {config['training']['base_model']}")

    adapter_paths = train_all_conditions(datasets, config)

    print("\nTraining complete. Adapter paths:")
    for name, path in adapter_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
