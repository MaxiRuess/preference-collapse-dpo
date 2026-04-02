"""PoliTune political data loading and formatting.

Loads left-leaning and right-leaning preference datasets from the PoliTune
project (scale-lab/politune-left, scale-lab/politune-right) and formats
them for SFT and DPO training conditions.

Data sources:
  - scale-lab/politune-right: 2,831 DPO pairs (right-chosen)
  - scale-lab/politune-left: 2,360 DPO pairs (left-chosen)
"""

from __future__ import annotations

import random

from datasets import Dataset, DatasetDict, load_dataset


def load_politune_datasets() -> tuple[Dataset, Dataset]:
    """Load left and right PoliTune datasets from HuggingFace Hub.

    Returns:
        Tuple of (left_dataset, right_dataset).
    """
    left = load_dataset("scale-lab/politune-left", split="train")
    right = load_dataset("scale-lab/politune-right", split="train")
    return left, right


def _to_sft_format(row: dict) -> dict:
    """Convert a PoliTune DPO row to SFT messages format."""
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
    }


def _to_dpo_format(row: dict) -> dict:
    """Convert a PoliTune row to conversational DPO format."""
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def _split_dataset(dataset: Dataset, train_ratio: float, seed: int) -> DatasetDict:
    """Split a dataset into train/eval by ratio."""
    split = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


# ---------------------------------------------------------------------------
# SFT dataset builders
# ---------------------------------------------------------------------------


def build_sft_right(right: Dataset, seed: int = 42) -> DatasetDict:
    """Build SFT dataset from right-leaning chosen responses."""
    sft_data = right.map(_to_sft_format, remove_columns=right.column_names)
    return _split_dataset(sft_data, train_ratio=0.9, seed=seed)


def build_sft_left(left: Dataset, seed: int = 42) -> DatasetDict:
    """Build SFT dataset from left-leaning chosen responses."""
    sft_data = left.map(_to_sft_format, remove_columns=left.column_names)
    return _split_dataset(sft_data, train_ratio=0.9, seed=seed)


def build_sft_merged(left: Dataset, right: Dataset, seed: int = 42) -> DatasetDict:
    """Build merged SFT dataset — 50/50 mix with randomly flipped labels.

    For each row, randomly decide whether the 'chosen' (original ideology)
    or 'rejected' (opposite ideology) is used as the SFT target.
    This creates contradictory training signal.
    """
    rng = random.Random(seed)
    rows = []

    for row in right:
        if rng.random() < 0.5:
            target = row["chosen"]  # right-leaning
        else:
            target = row["rejected"]  # left-leaning
        rows.append({
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": target},
            ]
        })

    for row in left:
        if rng.random() < 0.5:
            target = row["chosen"]  # left-leaning
        else:
            target = row["rejected"]  # right-leaning
        rows.append({
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": target},
            ]
        })

    rng.shuffle(rows)
    merged = Dataset.from_list(rows)
    return _split_dataset(merged, train_ratio=0.9, seed=seed)


# ---------------------------------------------------------------------------
# DPO dataset builders
# ---------------------------------------------------------------------------


def build_dpo_right(right: Dataset, seed: int = 42) -> DatasetDict:
    """Build DPO dataset from right-leaning preference pairs."""
    dpo_data = right.map(_to_dpo_format, remove_columns=right.column_names)
    return _split_dataset(dpo_data, train_ratio=0.9, seed=seed)


def build_dpo_left(left: Dataset, seed: int = 42) -> DatasetDict:
    """Build DPO dataset from left-leaning preference pairs."""
    dpo_data = left.map(_to_dpo_format, remove_columns=left.column_names)
    return _split_dataset(dpo_data, train_ratio=0.9, seed=seed)


def build_dpo_merged(left: Dataset, right: Dataset, seed: int = 42) -> DatasetDict:
    """Build merged DPO dataset — 50/50 contradictory preference labels.

    Pools all prompts from both datasets. For each prompt, randomly assigns
    whether the original chosen/rejected labels are kept or swapped.
    This creates contradictory preference signals.
    """
    rng = random.Random(seed)
    rows = []

    for row in right:
        dpo_row = _to_dpo_format(row)
        if rng.random() < 0.5:
            # Keep original: right chosen, left rejected
            rows.append(dpo_row)
        else:
            # Flip: left chosen, right rejected
            rows.append({
                "prompt": dpo_row["prompt"],
                "chosen": dpo_row["rejected"],
                "rejected": dpo_row["chosen"],
            })

    for row in left:
        dpo_row = _to_dpo_format(row)
        if rng.random() < 0.5:
            # Keep original: left chosen, right rejected
            rows.append(dpo_row)
        else:
            # Flip: right chosen, left rejected
            rows.append({
                "prompt": dpo_row["prompt"],
                "chosen": dpo_row["rejected"],
                "rejected": dpo_row["chosen"],
            })

    rng.shuffle(rows)
    merged = Dataset.from_list(rows)
    return _split_dataset(merged, train_ratio=0.9, seed=seed)


# ---------------------------------------------------------------------------
# Build all conditions
# ---------------------------------------------------------------------------


def build_all_politune_datasets(seed: int = 42) -> dict[str, DatasetDict]:
    """Build all experimental condition datasets from PoliTune data.

    Returns:
        Dict mapping condition name -> DatasetDict with train/eval splits.
    """
    print("Loading PoliTune datasets from HuggingFace Hub...")
    left, right = load_politune_datasets()
    print(f"  Left: {len(left)} pairs, Right: {len(right)} pairs")

    datasets = {}

    # SFT conditions
    print("Building SFT datasets...")
    datasets["sft_right"] = build_sft_right(right, seed)
    datasets["sft_left"] = build_sft_left(left, seed)
    datasets["sft_merged"] = build_sft_merged(left, right, seed)

    # DPO conditions
    print("Building DPO datasets...")
    datasets["dpo_right"] = build_dpo_right(right, seed)
    datasets["dpo_left"] = build_dpo_left(left, seed)
    datasets["dpo_merged"] = build_dpo_merged(left, right, seed)

    for name, ds in datasets.items():
        print(f"  {name}: train={len(ds['train'])}, eval={len(ds['eval'])}")

    return datasets
