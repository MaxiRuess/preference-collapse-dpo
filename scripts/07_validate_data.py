#!/usr/bin/env python3
"""Step 7: Validate the built PoliTune datasets before uploading to Modal.

Gate checks:
  1. No held-out prompt appears in ANY condition's train split.
  2. Left and right prompt sets do not overlap.
  3. Every condition's eval split contains only held-out prompts.
  4. Every Tier 5 evaluation prompt is held out of every train split.
  5. Reports duplicate-prompt and origin counts.

Exits non-zero on any failure.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from datasets import DatasetDict

from src.politune_data import ORIGIN_LEFT, ORIGIN_RIGHT, load_global_split


def _prompts(ds) -> list[str]:
    return [r["messages"][0]["content"] for r in ds]


def main():
    parser = argparse.ArgumentParser(description="Validate PoliTune datasets")
    parser.add_argument("--datasets-dir", default="data/politune_datasets")
    parser.add_argument("--n-per-origin", type=int, default=75,
                        help="Tier 5 prompts per origin (must match eval config)")
    args = parser.parse_args()

    root = Path(args.datasets_dir)
    split = load_global_split(root / "global_split.json")
    failures: list[str] = []

    conditions = sorted(p.name for p in root.iterdir()
                        if p.is_dir() and p.name.startswith("sft_"))
    print(f"Conditions: {conditions}")

    all_train: set[str] = set()
    for cond in conditions:
        ds = DatasetDict.load_from_disk(str(root / cond))
        train_p, eval_p = set(_prompts(ds["train"])), set(_prompts(ds["eval"]))
        all_train |= train_p
        leaked = train_p & split["eval"]
        if leaked:
            failures.append(f"{cond}: {len(leaked)} held-out prompts in train split")
        not_heldout = eval_p - split["eval"]
        if not_heldout:
            failures.append(f"{cond}: {len(not_heldout)} eval prompts not in global held-out set")
        if train_p & eval_p:
            failures.append(f"{cond}: {len(train_p & eval_p)} prompts in both splits")
        origins = Counter(r["origin"] for r in ds["train"])
        print(f"  {cond}: train={len(ds['train'])} eval={len(ds['eval'])} "
              f"train_origins={dict(origins)}")

    right_p = {p for p, o in split["origin"].items() if o == ORIGIN_RIGHT}
    left_p = {p for p, o in split["origin"].items() if o == ORIGIN_LEFT}
    if right_p & left_p:
        failures.append(f"{len(right_p & left_p)} prompts shared between left and right")

    # Tier 5 prompts as the evaluation code will sample them.
    from src.eval_prompts import load_eval_split_prompts
    tier5 = load_eval_split_prompts(datasets_dir=str(root), n_per_origin=args.n_per_origin)
    in_train = [p for p in tier5 if p["prompt"] in all_train]
    if in_train:
        failures.append(f"{len(in_train)}/{len(tier5)} Tier 5 prompts appear in some train split")
    print(f"  Tier 5: {len(tier5)} prompts, "
          f"{Counter(p['origin'] for p in tier5)}, in-train={len(in_train)}")

    s = split["stats"]
    print(f"\nRaw rows: {s['raw_rows']}  unique prompts: {s['unique_prompts']}  "
          f"duplicated prompts: {s['duplicate_prompts']}")
    print(f"Held-out prompts: {s['eval_prompts']}  train prompts: {s['train_prompts']}")

    if failures:
        print("\nVALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
