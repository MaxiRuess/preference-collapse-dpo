#!/usr/bin/env python3
"""Step 6: Evaluate all trained conditions on the four metrics."""

import argparse
import json

import yaml

from src.dataset_builder import load_datasets
from src.evaluation import run_full_evaluation, save_results
from src.generation import load_responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load eval data
    datasets = load_datasets(config["paths"]["datasets_dir"])
    consensus_pairs = load_responses(config["paths"]["consensus_file"])

    # Collect eval splits across conditions
    eval_data = []
    for name, ds in datasets.items():
        if "eval" in ds:
            for row in ds["eval"]:
                eval_data.append(row)

    # Map conditions to adapter paths (None = baseline)
    models_dir = config["paths"]["models_dir"]
    model_paths = {
        "baseline": None,
        "dpo_optimist": f"{models_dir}/dpo_optimist",
        "dpo_skeptic": f"{models_dir}/dpo_skeptic",
        "dpo_merged": f"{models_dir}/dpo_merged",
        "dpo_multi_optimist": f"{models_dir}/dpo_multi_optimist",
        "dpo_multi_skeptic": f"{models_dir}/dpo_multi_skeptic",
    }

    print(f"Evaluating {len(model_paths)} conditions on {len(eval_data)} eval prompts...")
    results = run_full_evaluation(model_paths, eval_data, consensus_pairs, config)

    output_path = config["paths"]["eval_results_file"]
    save_results(results, output_path)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for condition, metrics in results.items():
        print(f"\n{condition}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
