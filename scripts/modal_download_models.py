#!/usr/bin/env python3
"""Download trained adapters from Modal volume to local models/ directory.

Usage:
    python scripts/modal_download_models.py
    python scripts/modal_download_models.py --condition dpo_optimist
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download trained adapters from Modal")
    parser.add_argument("--condition", default=None, help="Download a specific condition (default: all)")
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if args.condition:
        # Download a single condition
        remote_path = f"{args.condition}/"
        local_path = str(models_dir / args.condition)
        Path(local_path).mkdir(exist_ok=True)
        cmd = [
            "modal", "volume", "get",
            "preference-collapse-models",
            remote_path,
            local_path,
        ]
    else:
        # Download everything
        cmd = [
            "modal", "volume", "get",
            "preference-collapse-models",
            "/",
            str(models_dir),
        ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    print(result.stdout)

    # List downloaded adapters
    adapters = [d.name for d in models_dir.iterdir() if (d / "adapter_config.json").exists()]
    print(f"Downloaded adapters: {adapters}")


if __name__ == "__main__":
    main()
