"""Visualization — ideology scores, Pareto frontier, consistency analysis.

Generates publication-quality figures for the preference collapse experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Color scheme: blue=left, red=right, purple=merged, gray=baseline
COLORS = {
    "baseline": "#808080",
    "sft_left": "#2166ac",
    "sft_right": "#b2182b",
    "sft_merged": "#7b3294",
    "merged_linear": "#e08214",
    "merged_ties": "#fdb863",
}

CONDITION_ORDER = [
    "baseline", "sft_left", "sft_right", "sft_merged",
    "merged_linear", "merged_ties",
]


def _ordered_conditions(stats: dict) -> list[str]:
    """Return conditions in display order, filtering to those present."""
    return [c for c in CONDITION_ORDER if c in stats]


def plot_ideology_scores(condition_stats: dict, output_path: str | Path) -> None:
    """Bar chart of mean ideology score per condition with 95% CI error bars."""
    conditions = _ordered_conditions(condition_stats)
    means = [condition_stats[c]["mean"] for c in conditions]
    cis = [condition_stats[c]["ci_95"] for c in conditions]
    bar_colors = [COLORS.get(c, "#808080") for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(conditions)), means, yerr=cis,
           color=bar_colors, edgecolor="black", linewidth=0.5,
           capsize=4, error_kw={"linewidth": 1.5})

    ax.axhline(y=10, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Center (10)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=10)
    ax.set_ylabel("Mean Ideology Score (0=Left, 20=Right)", fontsize=12)
    ax.set_title("Ideology Scores by Training Condition", fontsize=14)
    ax.set_ylim(0, 20)
    ax.legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pareto(condition_stats: dict, output_path: str | Path) -> None:
    """2D scatter: x=mean ideology score, y=consistency (1/std)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for condition, stats in condition_stats.items():
        x = stats["mean"]
        y = 1.0 / (stats["std"] + 0.01)
        color = COLORS.get(condition, "#808080")
        ax.scatter(x, y, s=120, c=color, edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.annotate(condition.replace("_", "\n"), (x, y),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)

    ax.axvline(x=10, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Mean Ideology Score (0=Left, 20=Right)", fontsize=12)
    ax.set_ylabel("Consistency (1 / std dev)", fontsize=12)
    ax.set_title("Pareto Frontier: Ideology vs Consistency", fontsize=14)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_consistency(consistency: dict, output_path: str | Path) -> None:
    """Grouped bar chart: within-topic std dev per condition."""
    conditions = [c for c in CONDITION_ORDER if c in consistency]
    topics = sorted(
        set(t for c in conditions for t in consistency[c].get("per_topic", {}))
    )
    if not topics:
        print("No consistency data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(topics))
    width = 0.8 / len(conditions)

    for i, condition in enumerate(conditions):
        stds = [consistency[condition]["per_topic"].get(t, 0) for t in topics]
        color = COLORS.get(condition, "#808080")
        ax.bar(x + i * width, stds, width, label=condition,
               color=color, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * len(conditions) / 2)
    ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Within-Topic Std Dev (lower = more consistent)", fontsize=11)
    ax.set_title("Ideological Consistency Across Paraphrased Prompts", fontsize=14)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tier_comparison(condition_stats: dict, output_path: str | Path) -> None:
    """Side-by-side bar charts for novel/adjacent/PoliTune tiers.

    Key memorization analysis: if ideology scores differ between novel
    and adjacent tiers, it suggests memorization rather than generalization.
    """
    tiers = ["novel", "adjacent", "politune"]
    conditions = _ordered_conditions(condition_stats)

    fig, axes = plt.subplots(1, len(tiers), figsize=(15, 5), sharey=True)

    for ax, tier in zip(axes, tiers):
        means = []
        bar_colors = []
        labels = []
        for cond in conditions:
            tier_data = condition_stats[cond].get("per_tier", {}).get(tier)
            if tier_data:
                means.append(tier_data["mean"])
                bar_colors.append(COLORS.get(cond, "#808080"))
                labels.append(cond)

        if means:
            ax.bar(range(len(means)), means, color=bar_colors,
                   edgecolor="black", linewidth=0.3)
            ax.axhline(y=10, color="black", linestyle="--",
                       linewidth=0.5, alpha=0.5)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=8)
        ax.set_title(f"Tier: {tier.capitalize()}", fontsize=12)
        ax.set_ylim(0, 20)

    axes[0].set_ylabel("Mean Ideology Score")
    fig.suptitle("Ideology Scores by Evaluation Tier (Memorization Analysis)",
                 fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_judge_agreement(
    generations: list[dict],
    judge_a: str,
    judge_b: str,
    output_path: str | Path,
) -> None:
    """Scatter plot of judge A scores vs judge B scores, colored by condition."""
    pairs = []
    for gen in generations:
        scores = gen.get("scores", {})
        sa = scores.get(judge_a)
        sb = scores.get(judge_b)
        if sa is not None and sb is not None:
            pairs.append((sa, sb, gen["condition"]))

    if not pairs:
        print("No paired scores to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for cond in CONDITION_ORDER:
        cond_pairs = [(a, b) for a, b, c in pairs if c == cond]
        if cond_pairs:
            xs = [p[0] for p in cond_pairs]
            ys = [p[1] for p in cond_pairs]
            color = COLORS.get(cond, "#808080")
            ax.scatter(xs, ys, c=color, label=cond, alpha=0.4, s=20, edgecolors="none")

    ax.plot([0, 20], [0, 20], "k--", linewidth=0.8, alpha=0.5, label="Perfect agreement")
    ax.set_xlabel(f"{judge_a} Score", fontsize=12)
    ax.set_ylabel(f"{judge_b} Score", fontsize=12)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Inter-Judge Agreement", fontsize=14)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_topic_heatmap(
    generations: list[dict],
    output_path: str | Path,
    tier_filter: str | None = None,
) -> None:
    """Heatmap of ideology scores per topic per condition."""
    from collections import defaultdict

    by_cond_topic = defaultdict(lambda: defaultdict(list))
    for r in generations:
        if r.get("score") is not None:
            if tier_filter and r.get("tier") != tier_filter:
                continue
            by_cond_topic[r["condition"]][r["topic"]].append(r["score"])

    conditions = [c for c in CONDITION_ORDER if c in by_cond_topic]
    topics = sorted(set(
        t for c in conditions for t in by_cond_topic[c]
    ))

    matrix = []
    for c in conditions:
        row = []
        for t in topics:
            scores = by_cond_topic[c][t]
            row.append(np.mean(scores) if scores else 10.0)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(14, len(topics) * 0.8), len(conditions) * 0.8 + 2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=20, aspect="auto")

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([t.replace("_", "\n") for t in topics],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=10)

    for i in range(len(conditions)):
        for j in range(len(topics)):
            val = matrix[i, j]
            color = "white" if val < 5 or val > 15 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Ideology Score (0=Left, 20=Right)", fontsize=10)

    title = "Per-Topic Ideology Scores"
    if tier_filter:
        title += f" (Tier: {tier_filter.capitalize()})"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_topic_bars(
    generations: list[dict],
    output_path: str | Path,
    topics: list[str] | None = None,
) -> None:
    """Grouped bar chart comparing merged conditions vs specialists per topic."""
    from collections import defaultdict

    by_cond_topic = defaultdict(lambda: defaultdict(list))
    for r in generations:
        if r.get("score") is not None:
            by_cond_topic[r["condition"]][r["topic"]].append(r["score"])

    focus_conditions = ["sft_left", "sft_right", "sft_merged", "merged_linear", "merged_ties"]
    conditions = [c for c in focus_conditions if c in by_cond_topic]

    if topics is None:
        topics = ["gun_control", "government_role", "economic_policy",
                  "social_justice", "environment"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(topics))
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        means = [np.mean(by_cond_topic[cond].get(t, [10])) for t in topics]
        color = COLORS.get(cond, "#808080")
        ax.bar(x + i * width, means, width, label=cond,
               color=color, edgecolor="black", linewidth=0.3)

    ax.axhline(y=10, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Center (10)")
    ax.set_xticks(x + width * len(conditions) / 2)
    ax.set_xticklabels(topics, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean Ideology Score (0=Left, 20=Right)", fontsize=11)
    ax.set_title("Per-Topic Ideology: Specialists vs Merged Conditions", fontsize=14)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
