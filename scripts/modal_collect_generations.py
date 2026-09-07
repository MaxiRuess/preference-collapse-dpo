#!/usr/bin/env python3
"""Collect per-instance generation files persisted on the Modal models volume
(generations_v2/<instance>.json) into data/eval_generations_v2.json.

Safety net for runs whose local client disconnected before results arrived.

Usage:
    python scripts/modal_collect_generations.py
    python scripts/modal_collect_generations.py --base-model gemma4
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.base_models import get_base_model
from src.generation import load_records, merge_rows, save_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="mistral")
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    spec = get_base_model(args.base_model)
    args.output_file = args.output_file or spec["generations_file"]

    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["modal", "volume", "get", "--force", "preference-collapse-models",
               f"{spec['volume_generations_dir']}/", tmp]
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr)
            sys.exit(1)
        files = sorted(Path(tmp).rglob("*.json"))
        rows = load_records(args.output_file)
        before = len(rows)
        for f in files:
            rows = merge_rows(rows, json.loads(f.read_text()))
            print(f"  {f.name}: total now {len(rows)}")
        save_records(args.output_file, rows)
        print(f"Added {len(rows) - before} records; {len(rows)} total in {args.output_file}")


if __name__ == "__main__":
    main()
