#!/usr/bin/env python3
"""Step 6: Evaluate all trained conditions.

Three-phase pipeline:
  1. Generate responses on Modal:  modal run modal_evaluate.py --condition all
  2. Score with LLM judge:         python scripts/06_evaluate.py --score
  3. Visualize results:            python scripts/06_evaluate.py --plot

Or run scoring + plotting:        python scripts/06_evaluate.py --all
"""

import argparse

import yaml

from src.evaluation import run_full_evaluation, load_results, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--generations", default="data/eval_generations.json",
                        help="Path to generated responses JSON (from modal_evaluate.py)")
    parser.add_argument("--results", default="data/eval_results.json",
                        help="Path to save/load evaluation results")
    parser.add_argument("--figures-dir", default="data/figures",
                        help="Directory for output figures")
    parser.add_argument("--score", action="store_true",
                        help="Score generations with LLM judge")
    parser.add_argument("--plot", action="store_true",
                        help="Generate visualizations from results")
    parser.add_argument("--all", action="store_true",
                        help="Run scoring + plotting")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.score or args.all:
        results = run_full_evaluation(args.generations, config, args.results)
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"{'Condition':<20} {'Mean':>6} {'Std':>6} {'CI95':>6} {'N':>4}")
        print("-" * 46)
        for cond, stats in sorted(results["condition_stats"].items()):
            print(f"{cond:<20} {stats['mean']:>6.1f} {stats['std']:>6.1f} "
                  f"{stats['ci_95']:>6.1f} {stats['n']:>4}")

        print(f"\nPareto frontier: {results['pareto']['frontier']}")
        print(f"Dominated:       {results['pareto']['dominated']}")

        if results.get("consistency"):
            print(f"\n{'Condition':<20} {'Mean Consistency Std':>20}")
            print("-" * 42)
            for cond, cons in sorted(results["consistency"].items()):
                val = cons.get("mean_within_topic_std")
                if val is not None:
                    print(f"{cond:<20} {val:>20.2f}")

    if args.plot or args.all:
        results = load_results(args.results)
        from src.visualization import (
            plot_ideology_scores, plot_pareto,
            plot_consistency, plot_tier_comparison,
        )
        fdir = args.figures_dir
        plot_ideology_scores(results["condition_stats"], f"{fdir}/ideology_scores.png")
        plot_pareto(results["condition_stats"], f"{fdir}/pareto.png")
        if results.get("consistency"):
            plot_consistency(results["consistency"], f"{fdir}/consistency.png")
        plot_tier_comparison(results["condition_stats"], f"{fdir}/tier_comparison.png")
        print(f"\nAll figures saved to {fdir}/")

    if not (args.score or args.plot or args.all):
        parser.print_help()
        print("\nHint: Run 'modal run modal_evaluate.py --condition all' first to generate responses.")


if __name__ == "__main__":
    main()
