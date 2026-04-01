"""Construct HuggingFace Datasets for each experimental condition.

Step 4 of the pipeline. Takes filtered response pairs and builds DPO-format
datasets (prompt, chosen, rejected) for each of the 7 conditions:

Core conditions (confident pairs only — clean ideological axis):
  - DPO-Optimist: optimist_confident chosen, skeptic_confident rejected
  - DPO-Skeptic: skeptic_confident chosen, optimist_confident rejected
  - DPO-Merged: 50/50 contradictory labels (confident pairs)
  - DPO-Multi: same as merged with "distribution" column for routing

Confidence asymmetry conditions (cross epistemic styles):
  - DPO-ConfOpt-UncSkp: 50/50 merged with optimist_confident vs skeptic_uncertain
  - DPO-ConfSkp-UncOpt: 50/50 merged with skeptic_confident vs optimist_uncertain

Baseline: no DPO dataset (base model used as-is)

Each dataset gets a 90/10 train/eval split with saved split metadata.
All conditions get the same number of training pairs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict


# --- Pair specifications per condition ---

# Core conditions use confident pairs only (clean ideological axis)
CONFIDENT_PAIRS = [
    ("optimist_confident", "skeptic_confident"),
]

# Confidence asymmetry: optimist confident vs skeptic uncertain
CONF_OPT_UNC_SKP_PAIRS = [
    ("optimist_confident", "skeptic_uncertain"),
]

# Confidence asymmetry: skeptic confident vs optimist uncertain
CONF_SKP_UNC_OPT_PAIRS = [
    ("skeptic_confident", "optimist_uncertain"),
]


def _pairs_from_responses(
    responses: list[dict],
    condition: str,
    seed: int,
) -> list[dict]:
    """Convert raw response records into DPO (prompt, chosen, rejected) triplets.

    Args:
        responses: List of response dicts from generation step.
        condition: One of the 6 training conditions.
        seed: Random seed for reproducible assignment in merged conditions.

    Returns:
        List of dicts with keys: prompt, prompt_id, chosen, rejected,
        and optionally "distribution" for dpo_multi.
    """
    rng = random.Random(seed)
    triplets = []

    for record in responses:
        prompt = record["prompt"]
        prompt_id = record["prompt_id"]
        resps = record["responses"]

        # Select which pair spec to use based on condition
        if condition in ("dpo_optimist", "dpo_skeptic", "dpo_merged", "dpo_multi"):
            pair_specs = CONFIDENT_PAIRS
        elif condition == "dpo_conf_opt_unc_skp":
            pair_specs = CONF_OPT_UNC_SKP_PAIRS
        elif condition == "dpo_conf_skp_unc_opt":
            pair_specs = CONF_SKP_UNC_OPT_PAIRS
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # For merged conditions: randomly pick which side is preferred
        if condition in ("dpo_merged", "dpo_multi",
                         "dpo_conf_opt_unc_skp", "dpo_conf_skp_unc_opt"):
            prefer_first = rng.random() < 0.5
        else:
            prefer_first = None

        for first_key, second_key in pair_specs:
            first_resp = resps[first_key]
            second_resp = resps[second_key]

            if condition == "dpo_optimist":
                chosen, rejected = first_resp, second_resp
            elif condition == "dpo_skeptic":
                chosen, rejected = second_resp, first_resp
            elif condition in ("dpo_merged", "dpo_multi"):
                if prefer_first:
                    chosen, rejected = first_resp, second_resp
                else:
                    chosen, rejected = second_resp, first_resp
            elif condition in ("dpo_conf_opt_unc_skp", "dpo_conf_skp_unc_opt"):
                # 50/50 contradictory labels, same as merged
                if prefer_first:
                    chosen, rejected = first_resp, second_resp
                else:
                    chosen, rejected = second_resp, first_resp
            else:
                raise ValueError(f"Unknown condition: {condition}")

            row = {
                "prompt_id": prompt_id,
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
            if condition == "dpo_multi":
                row["distribution"] = "optimist" if prefer_first else "skeptic"

            triplets.append(row)

    return triplets


def build_dpo_dataset(
    pairs: list[dict],
    condition: str,
    config: dict,
) -> DatasetDict:
    """Build a DPO dataset for a single experimental condition.

    Args:
        pairs: Filtered response pairs from the filtering step.
        condition: One of the 6 training conditions.
        config: Configuration dict (needs datasets section).

    Returns:
        A HuggingFace DatasetDict with "train" and "eval" splits.
    """
    ds_cfg = config["datasets"]
    seed = ds_cfg["seed"]

    triplets = _pairs_from_responses(pairs, condition, seed)

    # Split by prompt_id to avoid data leakage (all pairs from one prompt
    # go to the same split)
    prompt_ids = sorted(set(t["prompt_id"] for t in triplets))
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)

    n_train = int(len(prompt_ids) * ds_cfg["train_split"])
    train_ids = set(prompt_ids[:n_train])
    eval_ids = set(prompt_ids[n_train:])

    train_rows = [t for t in triplets if t["prompt_id"] in train_ids]
    eval_rows = [t for t in triplets if t["prompt_id"] in eval_ids]

    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "eval": Dataset.from_list(eval_rows),
    })


# All conditions to build
ALL_CONDITIONS = [
    "dpo_optimist",
    "dpo_skeptic",
    "dpo_merged",
    "dpo_multi",
    "dpo_conf_opt_unc_skp",
    "dpo_conf_skp_unc_opt",
]


def build_all_datasets(
    filtered_pairs: list[dict],
    config: dict,
) -> dict[str, DatasetDict]:
    """Build DPO datasets for all experimental conditions.

    Args:
        filtered_pairs: All filtered response pairs.
        config: Configuration dict.

    Returns:
        Dict mapping condition name -> DatasetDict.
    """
    datasets = {}

    for condition in ALL_CONDITIONS:
        datasets[condition] = build_dpo_dataset(filtered_pairs, condition, config)

    # Verify equal training sizes
    sizes = {name: len(ds["train"]) for name, ds in datasets.items()}
    assert len(set(sizes.values())) == 1, (
        f"Training set sizes must be equal across conditions, got: {sizes}"
    )

    return datasets


def save_datasets(datasets: dict[str, DatasetDict], output_dir: str | Path) -> None:
    """Save all datasets to disk with split metadata.

    Args:
        datasets: Dict mapping condition name -> DatasetDict.
        output_dir: Directory to save to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    for name, ds in datasets.items():
        ds_path = output_dir / name
        ds.save_to_disk(str(ds_path))

        metadata[name] = {
            "train_prompt_ids": sorted(set(ds["train"]["prompt_id"])),
            "eval_prompt_ids": sorted(set(ds["eval"]["prompt_id"])),
            "train_size": len(ds["train"]),
            "eval_size": len(ds["eval"]),
        }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_datasets(output_dir: str | Path) -> dict[str, DatasetDict]:
    """Load all datasets from disk.

    Args:
        output_dir: Directory containing saved datasets.

    Returns:
        Dict mapping condition name -> DatasetDict.
    """
    output_dir = Path(output_dir)
    datasets = {}

    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir() and (sub / "train").exists():
            datasets[sub.name] = DatasetDict.load_from_disk(str(sub))

    return datasets
