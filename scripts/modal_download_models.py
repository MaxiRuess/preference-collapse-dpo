#!/usr/bin/env python3
"""Download trained LoRA adapters from the Modal models volume to models/.

Usage:
    python scripts/modal_download_models.py                  # every *_adapter dir
    python scripts/modal_download_models.py --run sft_left_s42
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _volume_ls() -> list[str]:
    import json
    out = subprocess.run(["modal", "volume", "ls", "--json", "preference-collapse-models"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(f"Error: {out.stderr}")
        sys.exit(1)
    entries = json.loads(out.stdout)
    return sorted(e["Filename"] for e in entries
                  if e.get("Type") == "dir" and e["Filename"].endswith("_adapter"))


def main():
    parser = argparse.ArgumentParser(description="Download adapters from Modal")
    parser.add_argument("--run", default=None,
                        help="Run name, e.g. sft_left_s42 (default: all adapters)")
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    targets = [f"{args.run}_adapter"] if args.run else _volume_ls()
    if not targets:
        print("No *_adapter directories found on the volume.")
        sys.exit(1)

    for name in targets:
        local = models_dir / name
        if (local / "adapter_config.json").exists():
            print(f"Skipping {name} (already downloaded)")
            continue
        # Destination is the parent dir: modal creates models/<name>/ inside it.
        cmd = ["modal", "volume", "get", "--force", "preference-collapse-models",
               name, str(models_dir)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)

    adapters = sorted(d.name for d in models_dir.iterdir() if (d / "adapter_config.json").exists())
    print(f"Local adapters: {adapters}")


if __name__ == "__main__":
    main()
