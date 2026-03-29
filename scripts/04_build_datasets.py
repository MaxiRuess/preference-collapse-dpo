#!/usr/bin/env python3
"""Step 4: Build HuggingFace DPO datasets for each experimental condition."""

import argparse

import yaml

from src.dataset_builder import build_all_datasets, save_datasets
from src.generation import load_responses


def main():
    parser = argparse.ArgumentParser(description="Build DPO datasets")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    filtered_pairs = load_responses(config["paths"]["filtered_file"])
    print(f"Loaded {len(filtered_pairs)} filtered pairs")

    print("Building datasets for all conditions...")
    datasets = build_all_datasets(filtered_pairs, config)

    for name, ds in datasets.items():
        print(f"  {name}: train={len(ds['train'])}, eval={len(ds['eval'])}")

    output_dir = config["paths"]["datasets_dir"]
    save_datasets(datasets, output_dir)
    print(f"Saved datasets to {output_dir}")


if __name__ == "__main__":
    main()
