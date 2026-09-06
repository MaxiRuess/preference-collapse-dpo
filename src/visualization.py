"""Figures for the v2 evaluation (matplotlib, headless).

Entry point: ``make_all_figures(results, generations_path, out_dir)`` which
writes one PNG per figure for the primary judge under every protocol, plus a
judge-agreement scatter. Individual ``plot_*`` functions take the metric
dicts produced by src/evaluation.py.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

COLORS = {
    "baseline": "#808080",
    "sft_left": "#2166ac",
    "sft_right": "#b2182b",
    "sft_merged": "#7b3294",
    "merged_linear": "#e08214",
    "merged_ties": "#fdb863",
    "ctrl_linear_left": "#92c5de",
    "ctrl_linear_right": "#f4a582",
    "ctrl_ties_left": "#4393c3",
    "ctrl_ties_right": "#d6604d",
}

CONDITION_ORDER = [
    "baseline", "sft_left", "sft_right", "sft_merged", "merged_linear", "merged_ties",
    "ctrl_linear_left", "ctrl_linear_right", "ctrl_ties_left", "ctrl_ties_right",
]

MAIN_CONDITIONS = CONDITION_ORDER[:6]


def _ordered(keys) -> list[str]:
    known = [c for c in CONDITION_ORDER if c in keys]
    return known + sorted(k for k in keys if k not in CONDITION_ORDER)


def _label(c: str) -> str:
    return c.replace("_", "\n")


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Condition-level plots
# ---------------------------------------------------------------------------


def plot_condition_means(stats: dict, path, title: str, conditions=None) -> None:
    """Pooled mean with bootstrap CI; per-instance means overlaid as dots."""
    conds = [c for c in _ordered(stats) if stats[c].get("mean") is not None]
    if conditions:
        conds = [c for c in conds if c in conditions]
    means = [stats[c]["mean"] for c in conds]
    err = np.array([[stats[c]["mean"] - stats[c]["ci_95"][0], stats[c]["ci_95"][1] - stats[c]["mean"]]
                    for c in conds]).T
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(conds)), 5.5))
    ax.bar(range(len(conds)), means, yerr=err, capsize=4,
           color=[COLORS.get(c, "#999") for c in conds], edgecolor="black", linewidth=0.5)
    for i, c in enumerate(conds):
        inst = [v["mean"] for v in stats[c]["per_instance"].values() if v["mean"] is not None]
        ax.scatter(np.full(len(inst), i) + np.linspace(-0.15, 0.15, len(inst)), inst,
                   color="black", s=14, zorder=3)
    ax.axhline(10, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_label(c) for c in conds], fontsize=9)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Ideology score (0 = left, 20 = right)")
    ax.set_title(title)
    _save(fig, path)


def plot_pareto(pareto: dict, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond, p in pareto["points"].items():
        size = 80 + 300 * (p.get("hedge_rate") or 0)
        ax.scatter(p["mean"], p["consistency"], s=size, c=COLORS.get(cond, "#999"),
                   edgecolors="black", linewidths=0.5, zorder=3,
                   marker="o" if cond in pareto["frontier"] else "s")
        ax.annotate(cond, (p["mean"], p["consistency"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    ax.axvline(10, color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Mean ideology score (0 = left, 20 = right)")
    ax.set_ylabel("Consistency 1 / (SD + 0.01)")
    ax.set_title(f"{title}\n(circle = frontier, square = dominated; marker size ~ hedge rate)")
    _save(fig, path)


def plot_consistency(consistency: dict, path, title: str) -> None:
    """Within-topic SD per instance grouped by condition, with within-prompt SD for reference."""
    by_cond = defaultdict(list)
    for inst, v in consistency.items():
        if v.get("mean_within_topic_std") is not None:
            by_cond[v["condition"]].append(v)
    conds = _ordered(by_cond)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(conds)), 5.5))
    for i, c in enumerate(conds):
        vals = [v["mean_within_topic_std"] for v in by_cond[c]]
        noise = [v["mean_within_prompt_std"] for v in by_cond[c] if v.get("mean_within_prompt_std") is not None]
        ax.bar(i, np.mean(vals), color=COLORS.get(c, "#999"), edgecolor="black", linewidth=0.5,
               yerr=np.std(vals) if len(vals) > 1 else 0, capsize=4)
        if noise:
            ax.scatter([i], [np.mean(noise)], marker="_", s=300, color="black", zorder=3,
                       label="within-prompt SD (sampling noise)" if i == 0 else None)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_label(c) for c in conds], fontsize=9)
    ax.set_ylabel("Mean within-topic SD across paraphrases (lower = more consistent)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_variance_decomposition(var_decomp: dict, path, title: str) -> None:
    """Stacked bars: within-prompt vs between-prompt variance per instance."""
    insts = sorted(var_decomp, key=lambda i: (CONDITION_ORDER.index(var_decomp[i]["condition"])
                                              if var_decomp[i]["condition"] in CONDITION_ORDER else 99, i))
    within = [var_decomp[i]["within_prompt_var"] for i in insts]
    between = [var_decomp[i]["between_prompt_var"] for i in insts]
    colors = [COLORS.get(var_decomp[i]["condition"], "#999") for i in insts]
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(insts)), 5.5))
    x = np.arange(len(insts))
    ax.bar(x, within, color=colors, edgecolor="black", linewidth=0.4, hatch="//", label="within-prompt (sampling)")
    ax.bar(x, between, bottom=within, color=colors, edgecolor="black", linewidth=0.4, label="between-prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(insts, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Score variance")
    ax.set_title(f"{title}\n(hatched = coin-flip variance across samples of the same prompt)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_tier5_by_origin(t5: dict, path, title: str) -> None:
    conds = _ordered(t5)
    origins = sorted({o for c in conds for o in t5[c] if isinstance(t5[c][o], dict)})
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(conds)), 5.5))
    width = 0.8 / max(1, len(origins))
    for j, o in enumerate(origins):
        vals = [t5[c].get(o, {}).get("mean") or 0 for c in conds]
        ax.bar(np.arange(len(conds)) + j * width, vals, width, label=f"{o} prompts",
               color=[COLORS.get(c, "#999") for c in conds], edgecolor="black", linewidth=0.4,
               alpha=0.55 + 0.45 * j / max(1, len(origins) - 1))
    ax.axhline(10, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(np.arange(len(conds)) + width * (len(origins) - 1) / 2)
    ax.set_xticklabels([_label(c) for c in conds], fontsize=9)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Mean ideology score on Tier 5")
    ax.set_title(f"{title}\n(Tier 5 prompts are stance instructions; a gap reflects obedience)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_score_histograms(gens: list[dict], key: str, path, title: str,
                          tiers=("novel", "adjacent", "politune", "consistency")) -> None:
    from src.evaluation import get_score
    by_cond = defaultdict(list)
    for g in gens:
        if g.get("tier") in tiers:
            s = get_score(g, key)
            if s is not None:
                by_cond[g["condition"]].append(s)
    conds = [c for c in _ordered(by_cond) if c in MAIN_CONDITIONS] or _ordered(by_cond)
    fig, axes = plt.subplots(1, len(conds), figsize=(2.6 * len(conds), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        ax.hist(by_cond[c], bins=np.arange(-0.5, 21.5, 1), color=COLORS.get(c, "#999"),
                edgecolor="black", linewidth=0.3)
        ax.axvline(10, color="black", linestyle="--", linewidth=0.6)
        ax.set_title(c, fontsize=9)
        ax.set_xlim(-0.5, 20.5)
    axes[0].set_ylabel("count")
    fig.suptitle(title)
    _save(fig, path)


def plot_topic_heatmap(gens: list[dict], key: str, path, title: str, tier: str = "consistency") -> None:
    """Per-topic mean of per-prompt means, one row per instance."""
    from src.evaluation import get_score
    acc = defaultdict(lambda: defaultdict(list))
    for g in gens:
        if g.get("tier") == tier:
            s = get_score(g, key)
            if s is not None:
                acc[g["instance"]][g["topic"]].append(s)
    insts = sorted(acc)
    topics = sorted({t for i in insts for t in acc[i]})
    mat = np.array([[np.mean(acc[i][t]) if acc[i][t] else np.nan for t in topics] for i in insts])
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(topics)), 0.4 * len(insts) + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=20, aspect="auto")
    ax.set_xticks(range(len(topics))); ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(insts))); ax.set_yticklabels(insts, fontsize=7)
    for i in range(len(insts)):
        for j in range(len(topics)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] < 5 or mat[i, j] > 15 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("Ideology score")
    ax.set_title(title)
    _save(fig, path)


def plot_judge_agreement(gens: list[dict], key_a: str, key_b: str, path) -> None:
    from src.evaluation import get_score
    pts = defaultdict(list)
    for g in gens:
        a, b = get_score(g, key_a), get_score(g, key_b)
        if a is not None and b is not None:
            pts[g["condition"]].append((a, b))
    fig, ax = plt.subplots(figsize=(7, 7))
    for c in _ordered(pts):
        xs, ys = zip(*pts[c])
        ax.scatter(np.array(xs) + np.random.uniform(-0.2, 0.2, len(xs)),
                   np.array(ys) + np.random.uniform(-0.2, 0.2, len(ys)),
                   s=8, alpha=0.35, c=COLORS.get(c, "#999"), label=c, edgecolors="none")
    ax.plot([0, 20], [0, 20], "k--", linewidth=0.8)
    ax.set_xlabel(key_a); ax.set_ylabel(key_b)
    ax.set_xlim(-0.5, 20.5); ax.set_ylim(-0.5, 20.5); ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Inter-judge agreement")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_all_figures(results: dict, generations_path: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    gens = json.loads(Path(generations_path).read_text())
    primary = results["primary_judge"]
    for proto in results["protocols"]:
        key = f"{proto}/{primary}"
        r = results["by_key"].get(key)
        if not r:
            continue
        tag = f"{proto}_{primary}".replace("/", "_")
        t = f"{proto} protocol, judge {primary}"
        plot_condition_means(r["question_stats"], out / f"means_questions_{tag}.png",
                             f"Tiers 1-4 (neutral questions) — {t}", MAIN_CONDITIONS)
        plot_condition_means(r["question_stats"], out / f"means_controls_{tag}.png",
                             f"Same-ideology merge controls vs specialists — {t}",
                             ["sft_left", "sft_right", "ctrl_linear_left", "ctrl_linear_right",
                              "ctrl_ties_left", "ctrl_ties_right"])
        plot_pareto(r["pareto"], out / f"pareto_{tag}.png", f"Pareto frontier (Tiers 1-4) — {t}")
        plot_consistency(r["consistency"], out / f"consistency_{tag}.png", f"Paraphrase consistency (Tier 4) — {t}")
        plot_variance_decomposition(r["variance_decomposition"], out / f"variance_decomposition_{tag}.png",
                                    f"Variance decomposition (Tiers 1-4) — {t}")
        plot_tier5_by_origin(r["tier5_by_origin"], out / f"tier5_by_origin_{tag}.png", f"Tier 5 by prompt origin — {t}")
        plot_score_histograms(gens, key, out / f"histograms_{tag}.png", f"Score distributions, Tiers 1-4 — {t}")
        plot_topic_heatmap(gens, key, out / f"topic_heatmap_{tag}.png", f"Per-topic scores (Tier 4) — {t}")
    for proto in results["protocols"]:
        for j in results["judges"]:
            if j != primary:
                plot_judge_agreement(gens, f"{proto}/{primary}", f"{proto}/{j}",
                                     out / f"agreement_{proto}_{primary}_vs_{j}.png")
