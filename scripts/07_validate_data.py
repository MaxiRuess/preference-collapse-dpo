#!/usr/bin/env python3
"""Data validation — run BEFORE training to verify preference distributions.

Two checks:
  1. Inter-annotator agreement simulation: score 100 pairs from both
     distributions' perspectives, compute Cohen's kappa.
  2. Embedding analysis: embed responses, visualize with t-SNE/UMAP to confirm
     chosen responses from each distribution occupy distinct regions.
"""

import argparse

import yaml

from src.generation import load_responses


def compute_inter_annotator_agreement(pairs: list[dict], config: dict) -> dict:
    """Score pairs from both distributions and compute Cohen's kappa.

    Takes a sample of pairs, scores each from the optimist and skeptic
    perspectives, and measures within-distribution agreement (should be high)
    and between-distribution agreement (should be low).

    Args:
        pairs: Sample of response pairs.
        config: Configuration dict.

    Returns:
        Dict with kappa_within_optimist, kappa_within_skeptic,
        kappa_between, and raw scores.
    """
    raise NotImplementedError


def compute_embedding_analysis(pairs: list[dict], config: dict) -> dict:
    """Embed responses and compute separation metrics.

    Args:
        pairs: Response pairs to analyze.
        config: Configuration dict (needs validation section).

    Returns:
        Dict with embeddings, labels, and separation metrics.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Validate preference data before training")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = load_responses(config["paths"]["filtered_file"])
    sample_size = config["validation"]["sample_size"]
    sample = pairs[:sample_size]

    print(f"Running validation on {len(sample)} pairs...")

    # Inter-annotator agreement
    print("\n--- Inter-annotator agreement ---")
    agreement = compute_inter_annotator_agreement(sample, config)
    print(f"Within-optimist kappa:  {agreement.get('kappa_within_optimist', 'N/A')}")
    print(f"Within-skeptic kappa:   {agreement.get('kappa_within_skeptic', 'N/A')}")
    print(f"Between-distribution:   {agreement.get('kappa_between', 'N/A')}")

    # Embedding analysis
    if not args.skip_embeddings:
        from src.visualization import plot_embedding_analysis

        print("\n--- Embedding analysis ---")
        embedding_results = compute_embedding_analysis(pairs, config)
        output_path = f"{config['paths']['figures_dir']}/embedding_analysis.png"
        plot_embedding_analysis(
            embedding_results["embeddings"],
            embedding_results["labels"],
            output_path,
        )
        print(f"Embedding plot saved to {output_path}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
