"""PoliTune political data loading and dataset construction.

Loads the left-leaning and right-leaning preference datasets from the
PoliTune project (scale-lab/politune-left, scale-lab/politune-right) and
builds SFT datasets for the experimental conditions.

Data sources:
  - scale-lab/politune-right: right-chosen preference pairs (Truth Social prompts)
  - scale-lab/politune-left:  left-chosen preference pairs (Reddit prompts)

Split design (v2):
  A single *global prompt-level* split is made once over the union of unique
  prompts from both datasets, stratified by origin. Every condition
  (sft_right, sft_left, sft_merged_s*) uses the same held-out prompt set, so
  no evaluation prompt can appear in the training data of any model.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

ORIGIN_RIGHT = "truth_social"
ORIGIN_LEFT = "reddit"


def load_politune_datasets() -> tuple[Dataset, Dataset]:
    """Load left and right PoliTune datasets from the HuggingFace Hub."""
    left = load_dataset("scale-lab/politune-left", split="train")
    right = load_dataset("scale-lab/politune-right", split="train")
    return left, right


# ---------------------------------------------------------------------------
# Global prompt-level split
# ---------------------------------------------------------------------------


def make_global_prompt_split(
    left: Dataset,
    right: Dataset,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split the union of unique prompts once, stratified by origin.

    Returns a dict with:
      train:  set of prompts used for training (all conditions)
      eval:   set of held-out prompts (all conditions)
      origin: mapping prompt -> origin ("truth_social" | "reddit")
      stats:  counts used for reporting
    """
    right_unique = list(dict.fromkeys(right["prompt"]))
    left_unique = list(dict.fromkeys(left["prompt"]))
    overlap = set(right_unique) & set(left_unique)
    if overlap:
        raise ValueError(f"{len(overlap)} prompts appear in both datasets; "
                         "origin tagging would be ambiguous.")

    rng = random.Random(seed)
    train, evald, origin = set(), set(), {}
    for prompts, name in ((right_unique, ORIGIN_RIGHT), (left_unique, ORIGIN_LEFT)):
        shuffled = list(prompts)
        rng.shuffle(shuffled)
        n_eval = round(len(shuffled) * eval_ratio)
        evald.update(shuffled[:n_eval])
        train.update(shuffled[n_eval:])
        for p in prompts:
            origin[p] = name

    stats = {
        "seed": seed,
        "eval_ratio": eval_ratio,
        "raw_rows": {"right": len(right), "left": len(left)},
        "unique_prompts": {"right": len(right_unique), "left": len(left_unique)},
        "duplicate_prompts": {
            "right": sum(1 for v in Counter(right["prompt"]).values() if v > 1),
            "left": sum(1 for v in Counter(left["prompt"]).values() if v > 1),
        },
        "eval_prompts": {
            ORIGIN_RIGHT: sum(1 for p in evald if origin[p] == ORIGIN_RIGHT),
            ORIGIN_LEFT: sum(1 for p in evald if origin[p] == ORIGIN_LEFT),
        },
        "train_prompts": {
            ORIGIN_RIGHT: sum(1 for p in train if origin[p] == ORIGIN_RIGHT),
            ORIGIN_LEFT: sum(1 for p in train if origin[p] == ORIGIN_LEFT),
        },
    }
    return {"train": train, "eval": evald, "origin": origin, "stats": stats}


def save_global_split(split: dict, path: str | Path) -> None:
    """Persist the global split as JSON (one record per unique prompt)."""
    records = [
        {"prompt": p, "origin": o, "split": "eval" if p in split["eval"] else "train"}
        for p, o in split["origin"].items()
    ]
    payload = {"stats": split["stats"], "prompts": records}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_global_split(path: str | Path) -> dict:
    """Load a saved global split back into the in-memory format."""
    payload = json.loads(Path(path).read_text())
    train = {r["prompt"] for r in payload["prompts"] if r["split"] == "train"}
    evald = {r["prompt"] for r in payload["prompts"] if r["split"] == "eval"}
    origin = {r["prompt"]: r["origin"] for r in payload["prompts"]}
    return {"train": train, "eval": evald, "origin": origin, "stats": payload["stats"]}


# ---------------------------------------------------------------------------
# SFT dataset builders
# ---------------------------------------------------------------------------


def _sft_row(prompt: str, target: str, origin: str, target_ideology: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "origin": origin,
        "target_ideology": target_ideology,
    }


def _to_dataset_dict(rows: list[dict], split: dict, seed: int) -> DatasetDict:
    """Partition rows by the global split and shuffle the training rows."""
    train_rows = [r for r in rows if r["messages"][0]["content"] in split["train"]]
    eval_rows = [r for r in rows if r["messages"][0]["content"] in split["eval"]]
    assert len(train_rows) + len(eval_rows) == len(rows), "row lost in split"
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "eval": Dataset.from_list(eval_rows),
    })


def build_sft_right(right: Dataset, split: dict, seed: int = 42) -> DatasetDict:
    """SFT on right-leaning chosen responses."""
    rows = [_sft_row(r["prompt"], r["chosen"], ORIGIN_RIGHT, "right") for r in right]
    return _to_dataset_dict(rows, split, seed)


def build_sft_left(left: Dataset, split: dict, seed: int = 42) -> DatasetDict:
    """SFT on left-leaning chosen responses."""
    rows = [_sft_row(r["prompt"], r["chosen"], ORIGIN_LEFT, "left") for r in left]
    return _to_dataset_dict(rows, split, seed)


def build_sft_merged(left: Dataset, right: Dataset, split: dict, flip_seed: int) -> DatasetDict:
    """Merged SFT data: pooled prompts, target randomly chosen or rejected.

    For each row the SFT target is the ``chosen`` response (original ideology)
    with probability 0.5 and the ``rejected`` response (opposite ideology)
    otherwise. ``flip_seed`` controls the draw so that each training seed can
    see a different realisation of the label noise.
    """
    rng = random.Random(flip_seed)
    rows = []
    for r in right:
        if rng.random() < 0.5:
            rows.append(_sft_row(r["prompt"], r["chosen"], ORIGIN_RIGHT, "right"))
        else:
            rows.append(_sft_row(r["prompt"], r["rejected"], ORIGIN_RIGHT, "left"))
    for r in left:
        if rng.random() < 0.5:
            rows.append(_sft_row(r["prompt"], r["chosen"], ORIGIN_LEFT, "left"))
        else:
            rows.append(_sft_row(r["prompt"], r["rejected"], ORIGIN_LEFT, "right"))
    return _to_dataset_dict(rows, split, flip_seed)


# ---------------------------------------------------------------------------
# Build all conditions
# ---------------------------------------------------------------------------


def build_all_politune_datasets(
    seed: int = 42,
    flip_seeds: tuple[int, ...] = (42, 43, 44),
    eval_ratio: float = 0.1,
) -> tuple[dict[str, DatasetDict], dict]:
    """Build every SFT condition from one global prompt split.

    Returns (datasets, split) where datasets maps condition name ->
    DatasetDict and split is the global split dict.
    """
    print("Loading PoliTune datasets from HuggingFace Hub...")
    left, right = load_politune_datasets()
    print(f"  Left: {len(left)} rows, Right: {len(right)} rows")

    split = make_global_prompt_split(left, right, eval_ratio=eval_ratio, seed=seed)
    s = split["stats"]
    print(f"  Unique prompts: right={s['unique_prompts']['right']}, "
          f"left={s['unique_prompts']['left']}")
    print(f"  Held-out prompts: {s['eval_prompts']}")

    datasets: dict[str, DatasetDict] = {
        "sft_right": build_sft_right(right, split, seed),
        "sft_left": build_sft_left(left, split, seed),
    }
    for fs in flip_seeds:
        datasets[f"sft_merged_s{fs}"] = build_sft_merged(left, right, split, flip_seed=fs)

    for name, ds in datasets.items():
        print(f"  {name}: train={len(ds['train'])}, eval={len(ds['eval'])}")
    return datasets, split
