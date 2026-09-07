#!/usr/bin/env python3
"""Step 9: Adapter geometry (delta norms and cosine similarities) per base model.

Reproduces data/adapter_geometry_v2.json for any base model from the local
adapters (download with scripts/modal_download_models.py [--base-model TAG]).
Cosines are computed over the concatenation of every adapted module's dense
delta, accumulated module by module so nothing large is materialised.

Usage:
    python scripts/09_adapter_geometry.py                    # mistral -> data/adapter_geometry_v2.json
    python scripts/09_adapter_geometry.py --base-model gemma4  # -> data/adapter_geometry_v2_gemma4.json
"""

import argparse
import itertools
import json
from pathlib import Path

from src.base_models import get_base_model
from src.generation import load_adapter_deltas


def _dot(a: dict, b: dict) -> float:
    return float(sum((a[m].double() * b[m].double()).sum().item() for m in a))


def _norm(a: dict) -> float:
    return _dot(a, a) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="mistral")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    spec = get_base_model(args.base_model)
    root = Path(args.models_root) / spec["models_subdir"] if spec["models_subdir"] else Path(args.models_root)
    suffix = "" if spec["tag"] == "mistral" else f"_{spec['tag']}"
    out_path = args.output or f"data/adapter_geometry_v2{suffix}.json"
    seeds = spec["seeds"]

    deltas, norms = {}, {}
    for cond in ("sft_left", "sft_right", "sft_merged"):
        for s in seeds:
            name = f"{cond}_s{s}"
            path = root / f"{name}_adapter"
            if not (path / "adapter_config.json").exists():
                raise SystemExit(f"missing adapter {path}; run scripts/modal_download_models.py --base-model {spec['tag']}")
            deltas[name] = load_adapter_deltas(str(path))
            norms[name] = _norm(deltas[name])
            print(f"{name}: {len(deltas[name])} modules, ||dW|| = {norms[name]:.3f}")

    def cos(a, b):
        return _dot(deltas[a], deltas[b]) / (norms[a] * norms[b])

    geom = {
        "base_model": spec["tag"],
        "hf_id": spec["hf_id"],
        "n_modules": len(next(iter(deltas.values()))),
        "norms": norms,
        "cos_left_right_same_seed": [cos(f"sft_left_s{s}", f"sft_right_s{s}") for s in seeds],
        "cos_left_seeds": [cos(f"sft_left_s{a}", f"sft_left_s{b}") for a, b in itertools.combinations(seeds, 2)],
        "cos_right_seeds": [cos(f"sft_right_s{a}", f"sft_right_s{b}") for a, b in itertools.combinations(seeds, 2)],
        "cos_merged_seeds": [cos(f"sft_merged_s{a}", f"sft_merged_s{b}") for a, b in itertools.combinations(seeds, 2)],
        "cos_merged_vs_left": [cos(f"sft_merged_s{s}", f"sft_left_s{s}") for s in seeds],
        "cos_merged_vs_right": [cos(f"sft_merged_s{s}", f"sft_right_s{s}") for s in seeds],
    }
    Path(out_path).write_text(json.dumps(geom, indent=2))
    print(f"\nleft/right same seed cos: {[round(c, 3) for c in geom['cos_left_right_same_seed']]}")
    print(f"left seeds cos: {[round(c, 3) for c in geom['cos_left_seeds']]}   "
          f"right seeds cos: {[round(c, 3) for c in geom['cos_right_seeds']]}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
