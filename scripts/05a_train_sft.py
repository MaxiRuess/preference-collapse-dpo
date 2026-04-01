#!/usr/bin/env python3
"""Step 5a: Train unified SFT on UltraChat.

Produces a general-purpose chat model from the base model.
Run this ONCE before any DPO training.
"""

import argparse

import yaml

from src.sft_training import train_sft


def main():
    parser = argparse.ArgumentParser(description="Train unified SFT model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Base model: {config['training']['base_model']}")
    print("Training unified SFT on UltraChat...")

    sft_path = train_sft(config)
    print(f"\nSFT model saved to: {sft_path}")
    print("You can now run DPO training with: python scripts/05_train_dpo.py")


if __name__ == "__main__":
    main()
