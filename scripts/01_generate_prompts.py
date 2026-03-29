#!/usr/bin/env python3
"""Step 1: Save prompts to JSONL.

Exports the 694 hand-crafted prompts from src/prompts.py to a JSONL file
for use by downstream pipeline steps.
"""

import argparse

import yaml

from src.prompts import SEED_PROMPTS, save_prompts


def main():
    parser = argparse.ArgumentParser(description="Export prompts to JSONL")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loaded {len(SEED_PROMPTS)} prompts")

    output_path = config["paths"]["prompts_file"]
    save_prompts(SEED_PROMPTS, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
