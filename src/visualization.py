"""Visualization — Pareto plot, win rate charts, consistency, embeddings.

Generates publication-quality figures for the paper.
"""

from __future__ import annotations

from pathlib import Path


def plot_pareto(results: dict, output_path: str | Path) -> None:
    """Plot the Pareto frontier — the central figure of the paper.

    2D scatter: x = optimist rubric win rate, y = skeptic rubric win rate.
    Each point is an experimental condition. Draws the Pareto frontier line.

    Expected pattern:
      - DPO-Optimist: top-left
      - DPO-Skeptic: bottom-right
      - DPO-Merged: interior (dominated)
      - DPO-Multi: closer to frontier than Merged
      - Baseline: middle

    Args:
        results: Evaluation results dict (condition -> scores).
        output_path: Where to save the figure (PDF/PNG).
    """
    raise NotImplementedError


def plot_win_rates(results: dict, output_path: str | Path) -> None:
    """Bar chart of win rates per condition under both rubrics.

    Grouped bars: each condition gets two bars (optimist score, skeptic score).

    Args:
        results: Evaluation results dict.
        output_path: Where to save the figure.
    """
    raise NotImplementedError


def plot_consistency(results: dict, output_path: str | Path) -> None:
    """Bar chart comparing consistency scores across conditions.

    Args:
        results: Evaluation results dict.
        output_path: Where to save the figure.
    """
    raise NotImplementedError


def plot_embedding_analysis(
    embeddings: dict,
    labels: dict,
    output_path: str | Path,
) -> None:
    """t-SNE or UMAP visualization of response embeddings.

    Shows that chosen responses from Distribution A and B occupy distinct
    regions for contested prompts but overlap for consensus prompts.

    Args:
        embeddings: Dict mapping response_id -> embedding vector.
        labels: Dict mapping response_id -> metadata (distribution, contested/consensus).
        output_path: Where to save the figure.
    """
    raise NotImplementedError
