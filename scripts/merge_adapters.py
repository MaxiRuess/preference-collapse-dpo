#!/usr/bin/env python3
"""Local adapter merging (thin wrapper over src/generation.py).

Requires the seed adapters under models/ (download with
scripts/modal_download_models.py) and a GPU with ~16 GB for the bf16 base.

Usage:
    python scripts/merge_adapters.py --instance merged_linear_s42 --smoke
    python scripts/merge_adapters.py --instance merge,control --generate
"""

import argparse
from pathlib import Path

import yaml

from src.generation import (
    GEN_DEFAULTS, build_instances, generate_samples, load_instance_model,
    load_records, missing_prompts, save_records, select_instances,
)


def main():
    parser = argparse.ArgumentParser(description="Merge SFT adapters locally")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--instance", default="merge,control")
    parser.add_argument("--generate", action="store_true",
                        help="Generate eval responses into the v2 generations file")
    parser.add_argument("--smoke", action="store_true",
                        help="Merge and print one short response per instance")
    parser.add_argument("--output", default="data/eval_generations_v2.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    gen_cfg = {**GEN_DEFAULTS, **cfg.get("generation", {})}
    merge_cfg = {"ties_density": 0.5, **cfg.get("merging", {})}
    control_pairs = [tuple(p) for p in merge_cfg.get("control_pairs", [(42, 43)])]
    instances = [i for i in build_instances(cfg["training"]["seeds"], control_pairs, args.models_root)
                 if i["kind"] in ("merge", "control")]
    instances = select_instances(instances, args.instance)

    for inst in instances:
        missing = [a for a in inst["adapters"] if not Path(a, "adapter_config.json").exists()]
        if missing:
            print(f"Skipping {inst['instance']}: adapters not found {missing}")
            continue
        model, tokenizer = load_instance_model(inst, density=merge_cfg["ties_density"])

        if args.smoke:
            probe = [{"id": "probe", "prompt": "What role should the government play in healthcare?",
                      "tier": "probe", "topic": "probe"}]
            rows = generate_samples(model, tokenizer, probe, inst, samples_per_prompt=1,
                                    max_new_tokens=120, batch_prompts=1)
            print(f"\n[{inst['instance']}] {rows[0]['response'][:300]}\n")

        if args.generate:
            from src.eval_prompts import get_all_eval_prompts
            prompts = get_all_eval_prompts(n_per_origin=cfg["datasets"]["n_eval_split_per_origin"])
            rows = load_records(args.output)
            needed = missing_prompts(rows, inst, prompts, gen_cfg["samples_per_prompt"])
            if not needed:
                print(f"Skipping {inst['instance']} — complete")
            else:
                rows.extend(generate_samples(model, tokenizer, needed, inst, **gen_cfg))
                save_records(args.output, rows)
                print(f"Saved {len(rows)} records to {args.output}")

        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
