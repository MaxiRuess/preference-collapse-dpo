"""Evaluation — win rates, consistency, consensus preservation, Pareto analysis.

Step 6 of the pipeline. Four metrics:

  6a. Per-distribution win rate (LLM-as-judge against both rubrics)
  6b. Consistency score (paraphrase stability)
  6c. Consensus preservation (performance on non-controversial pairs)
  6d. Pareto analysis (2D optimist-score × skeptic-score frontier)
"""

from __future__ import annotations

import json
from pathlib import Path


def evaluate_win_rate(
    model_path: str | None,
    eval_data: list[dict],
    rubric: str,
    config: dict,
) -> float:
    """Compute win rate for a model under a given rubric.

    Generates responses from the model on eval prompts, then uses an LLM judge
    to score each response against the rubric.

    Args:
        model_path: Path to LoRA adapter (None for baseline).
        eval_data: List of eval prompt dicts.
        rubric: The evaluation rubric string (optimist or skeptic).
        config: Configuration dict.

    Returns:
        Mean score (0–1) across all eval prompts.
    """
    raise NotImplementedError


def evaluate_consistency(
    model_path: str | None,
    eval_prompts: list[str],
    config: dict,
) -> float:
    """Measure response consistency across paraphrased prompts.

    For each eval prompt, generates paraphrases, gets model responses to
    both original and paraphrases, and measures position consistency.

    Args:
        model_path: Path to LoRA adapter (None for baseline).
        eval_prompts: List of eval prompt strings.
        config: Configuration dict.

    Returns:
        Consistency score (0–1). Higher = more consistent positions.
    """
    raise NotImplementedError


def evaluate_consensus(
    model_path: str | None,
    consensus_pairs: list[dict],
    config: dict,
) -> float:
    """Evaluate performance on consensus (non-controversial) pairs.

    Tests whether the model can still identify clearly better responses
    when both distributions agree.

    Args:
        model_path: Path to LoRA adapter (None for baseline).
        consensus_pairs: List of consensus pair dicts.
        config: Configuration dict.

    Returns:
        Accuracy (0–1) on consensus pairs.
    """
    raise NotImplementedError


def compute_pareto_frontier(results: dict) -> dict:
    """Compute the Pareto frontier from per-condition results.

    Places each condition on a 2D plane:
      x-axis = win rate under optimist rubric
      y-axis = win rate under skeptic rubric

    Args:
        results: Dict mapping condition -> {optimist_score, skeptic_score, ...}.

    Returns:
        Dict with "frontier" (list of non-dominated conditions),
        "dominated" (list of dominated conditions), and "all_points".
    """
    raise NotImplementedError


def run_full_evaluation(
    model_paths: dict[str, str | None],
    eval_data: list[dict],
    consensus_pairs: list[dict],
    config: dict,
) -> dict:
    """Run all four evaluation metrics for all conditions.

    Args:
        model_paths: Dict mapping condition name -> adapter path (None for baseline).
        eval_data: List of eval prompt dicts.
        consensus_pairs: List of consensus pair dicts.
        config: Configuration dict.

    Returns:
        Nested dict: condition -> metric -> value.
    """
    raise NotImplementedError


def save_results(results: dict, path: str | Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str | Path) -> dict:
    """Load evaluation results from JSON."""
    with open(path) as f:
        return json.load(f)
