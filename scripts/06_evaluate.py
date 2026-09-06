#!/usr/bin/env python3
"""Step 6: Score generations with the judge panel and compute all metrics.

Pipeline:
  1. Generate responses on Modal:
       modal run modal_evaluate.py --instance all
       modal run modal_merge_adapters.py
  2. Dry run the judges (20 records, every judge/protocol):
       python scripts/06_evaluate.py --score --limit 20
  3. Full scoring + metrics:
       python scripts/06_evaluate.py --all
  4. Figures and LaTeX tables:
       python scripts/06_evaluate.py --plot
       python scripts/08_analysis_tables.py
"""

import argparse

import yaml

from src.evaluation import load_results, run_full_evaluation


def _print_summary(results: dict) -> None:
    primary = results["primary_judge"]
    for proto in results["protocols"]:
        key = f"{proto}/{primary}"
        if key not in results["by_key"]:
            continue
        r = results["by_key"][key]
        print(f"\n{'='*72}\nPROTOCOL {proto} | JUDGE {primary} | Tiers 1-4 (question prompts)\n{'='*72}")
        print(f"{'Condition':<22}{'Mean':>7}{'SD':>7}{'CI95':>16}{'Hedge':>7}{'N':>6}{'Inst':>5}")
        for cond, s in r["question_stats"].items():
            if s["mean"] is None:
                continue
            ci = f"[{s['ci_95'][0]:.1f},{s['ci_95'][1]:.1f}]"
            hedge = f"{s['hedge_rate']:.2f}" if s["hedge_rate"] is not None else "-"
            print(f"{cond:<22}{s['mean']:>7.2f}{s['std']:>7.2f}{ci:>16}{hedge:>7}{s['n_scored']:>6}{s['n_instances']:>5}")

        print(f"\nPareto frontier: {r['pareto']['frontier']}")
        print(f"Dominated:       {r['pareto']['dominated']}")

        print(f"\n{'Condition':<22}{'Within-topic SD':>16}{'Within-prompt share':>20}")
        for cond, c in r["consistency_by_condition"].items():
            share = r["within_prompt_share_by_condition"].get(cond, {}).get("mean")
            share_s = f"{share:.2f}" if share is not None else "-"
            print(f"{cond:<22}{c['mean']:>16.2f}{share_s:>20}")

        print("\nTier 5 by origin (mean score):")
        for cond, o in r["tier5_by_origin"].items():
            parts = [f"{k}={v['mean']:.1f}" for k, v in o.items() if isinstance(v, dict) and v["mean"] is not None]
            print(f"  {cond:<22}{'  '.join(parts)}  gap={o.get('origin_gap')}")

        print("\nBrown-Forsythe variance tests (per-prompt means, Tiers 1-4):")
        for name, t in r["variance_tests"].items():
            if "p_value" in t:
                print(f"  {name:<32} sd {t['sd_a']:.2f} vs {t['sd_b']:.2f}  p={t['p_value']:.4f}")

    for proto, agr in results.get("agreement", {}).items():
        print(f"\nInter-judge agreement ({proto}): alpha={agr['overall']['krippendorff_alpha']}")
        for pair, m in agr["overall"]["pairs"].items():
            if "pearson_r" in m:
                print(f"  {pair:<40} r={m['pearson_r']:.3f} kappa={m['cohens_kappa_5bin']:.3f} "
                      f"MAD={m['mean_abs_diff']:.2f} n={m['n']}")


def main():
    parser = argparse.ArgumentParser(description="Score and evaluate generations")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--generations", default=None,
                        help="Generations JSON (default: paths.generations_file)")
    parser.add_argument("--results", default=None,
                        help="Results JSON (default: paths.eval_results_file)")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--score", action="store_true", help="Run the judges")
    parser.add_argument("--metrics", action="store_true", help="Compute metrics from stored scores")
    parser.add_argument("--plot", action="store_true", help="Generate figures")
    parser.add_argument("--all", action="store_true", help="score + metrics + plot")
    parser.add_argument("--judge", default=None, help="Comma-separated judge names (default: all)")
    parser.add_argument("--protocol", default=None, help="Comma-separated protocols (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N records (dry run)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Write scores to the JSONL cache only (safe for parallel judge processes); "
                             "skip metrics")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    generations = args.generations or paths.get("generations_file", "data/eval_generations_v2.json")
    results_path = args.results or paths.get("eval_results_file", "data/eval_results_v2.json")
    figures_dir = args.figures_dir or paths.get("figures_dir", "data/figures_v2")
    judges = args.judge.split(",") if args.judge else None
    protocols = args.protocol.split(",") if args.protocol else None

    if args.cache_only:
        from src.evaluation import run_scoring
        run_scoring(generations, config, judges, protocols, args.limit, cache_only=True)
        return

    if args.score or args.metrics or args.all:
        results = run_full_evaluation(
            generations, config, results_path, judges=judges, protocols=protocols,
            score=bool(args.score or args.all), limit=args.limit,
        )
        _print_summary(results)

    if args.plot or args.all:
        results = load_results(results_path)
        from src.visualization import make_all_figures
        make_all_figures(results, generations, figures_dir)

    if not (args.score or args.metrics or args.plot or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()
