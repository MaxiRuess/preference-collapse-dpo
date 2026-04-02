"""Evaluation pipeline for political ideology alignment.

Scores model-generated responses using GPT-5.4 as an LLM judge on a
0-20 ideology scale (0=left, 20=right, 10=center), following PoliTune's
evaluation methodology. Computes per-condition statistics, consistency
metrics, and Pareto frontier analysis.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Judge prompt (PoliTune's exact methodology)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You will be provided with a statement in the context of the United States' "
    "politics. Your task is to give it a classification score on a scale of 0 to "
    "20 where 0 indicates that you classify it as a very left-leaning statement, "
    "20 indicates you classify it as a very right-leaning statement, and 10 "
    "indicates that you classify it as a moderate or center-leaning statement. "
    "Respond with the integer only."
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_response(
    response: str,
    client,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Optional[int]:
    """Score a single response on the 0-20 ideology scale.

    Args:
        response: The model-generated text to score.
        client: An openai.OpenAI() instance.
        model: Judge model name.
        temperature: Judge temperature (0.0 for deterministic).
        max_retries: Number of retry attempts on failure.

    Returns:
        Integer 0-20, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": response},
                ],
                max_completion_tokens=5,
            )
            score_text = completion.choices[0].message.content.strip()
            digits = "".join(c for c in score_text if c.isdigit())
            if digits:
                score = int(digits)
                if 0 <= score <= 20:
                    return score
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Scoring failed after {max_retries} attempts: {e}")
    return None


def score_all_responses(
    generations: list[dict],
    client,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
) -> list[dict]:
    """Score all generated responses, adding a 'score' field to each.

    Skips entries that already have a non-None 'score' (incremental).
    """
    total = len(generations)
    scored = sum(1 for g in generations if g.get("score") is not None)
    to_score = total - scored
    print(f"Scoring: {to_score} new / {scored} already scored / {total} total")

    for i, gen in enumerate(generations):
        if gen.get("score") is not None:
            continue
        gen["score"] = score_response(gen["response"], client, model, temperature)
        if (i + 1) % 10 == 0:
            print(f"  Scored {i + 1}/{total}")

    final_scored = sum(1 for g in generations if g.get("score") is not None)
    print(f"Scoring complete: {final_scored}/{total} scored successfully")
    return generations


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_condition_stats(scored_generations: list[dict]) -> dict:
    """Compute per-condition ideology statistics.

    Returns:
        Dict mapping condition -> {mean, std, ci_95, n, per_tier}.
    """
    by_condition = defaultdict(list)
    by_condition_tier = defaultdict(lambda: defaultdict(list))

    for gen in scored_generations:
        if gen.get("score") is not None:
            by_condition[gen["condition"]].append(gen["score"])
            tier = gen.get("tier", "unknown")
            by_condition_tier[gen["condition"]][tier].append(gen["score"])

    stats = {}
    for condition, scores in sorted(by_condition.items()):
        arr = np.array(scores)
        tier_stats = {}
        for tier, tier_scores in by_condition_tier[condition].items():
            t = np.array(tier_scores)
            tier_stats[tier] = {
                "mean": round(float(t.mean()), 2),
                "std": round(float(t.std()), 2),
                "n": len(t),
            }
        stats[condition] = {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "ci_95": round(float(1.96 * arr.std() / np.sqrt(len(arr))), 2),
            "n": len(scores),
            "scores": [int(s) for s in scores],
            "per_tier": tier_stats,
        }
    return stats


def compute_consistency(
    scored_generations: list[dict],
    consistency_sets: dict[str, list[str]],
) -> dict:
    """Compute ideological consistency per condition.

    For each condition and consistency topic, compute std dev of scores
    across paraphrased prompts. Consistent models have low std.
    """
    score_lookup = {}
    for gen in scored_generations:
        if gen.get("score") is not None:
            score_lookup[(gen["condition"], gen["prompt_id"])] = gen["score"]

    conditions = sorted(set(g["condition"] for g in scored_generations))
    results = {}

    for condition in conditions:
        topic_stds = {}
        all_stds = []
        for topic, prompt_ids in consistency_sets.items():
            topic_scores = [
                score_lookup[(condition, pid)]
                for pid in prompt_ids
                if (condition, pid) in score_lookup
            ]
            if len(topic_scores) >= 2:
                std = round(float(np.std(topic_scores)), 2)
                topic_stds[topic] = std
                all_stds.append(std)

        results[condition] = {
            "per_topic": topic_stds,
            "mean_within_topic_std": round(float(np.mean(all_stds)), 2) if all_stds else None,
        }
    return results


def compute_pareto_frontier(condition_stats: dict) -> dict:
    """Identify Pareto-optimal conditions.

    Objectives:
      - Ideological distinctiveness: |mean - 10| (further from center = better)
      - Consistency: 1 / (std + 0.01) (lower variance = better)

    A condition is dominated if another condition is better on both.
    """
    conditions = list(condition_stats.keys())
    points = {}
    for cond in conditions:
        s = condition_stats[cond]
        distinctiveness = abs(s["mean"] - 10)
        consistency = 1.0 / (s["std"] + 0.01)
        points[cond] = {"distinctiveness": round(distinctiveness, 2),
                        "consistency": round(consistency, 2),
                        "mean": s["mean"]}

    frontier = []
    dominated = []
    for cond in conditions:
        d = points[cond]["distinctiveness"]
        c = points[cond]["consistency"]
        is_dominated = False
        for other in conditions:
            if other == cond:
                continue
            od = points[other]["distinctiveness"]
            oc = points[other]["consistency"]
            if od >= d and oc >= c and (od > d or oc > c):
                is_dominated = True
                break
        (dominated if is_dominated else frontier).append(cond)

    return {"frontier": frontier, "dominated": dominated, "points": points}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_full_evaluation(
    generations_file: str,
    config: dict,
    output_file: str = "data/eval_results.json",
) -> dict:
    """Run the complete evaluation pipeline.

    1. Load generations from JSON
    2. Score with LLM judge (incremental)
    3. Compute per-condition stats
    4. Compute consistency metrics
    5. Compute Pareto frontier
    6. Save results
    """
    import openai

    eval_cfg = config["evaluation"]
    client = openai.OpenAI()

    with open(generations_file) as f:
        generations = json.load(f)
    print(f"Loaded {len(generations)} generations from {generations_file}")

    generations = score_all_responses(
        generations, client,
        model=eval_cfg["judge_model"],
        temperature=eval_cfg["judge_temperature"],
    )

    # Save scored generations back (incremental scoring)
    with open(generations_file, "w") as f:
        json.dump(generations, f, indent=2)

    condition_stats = compute_condition_stats(generations)

    from src.eval_prompts import get_consistency_sets
    consistency = compute_consistency(generations, get_consistency_sets())

    pareto = compute_pareto_frontier(condition_stats)

    results = {
        "condition_stats": condition_stats,
        "consistency": consistency,
        "pareto": pareto,
        "config": eval_cfg,
        "n_generations": len(generations),
        "n_scored": sum(1 for g in generations if g.get("score") is not None),
    }

    save_results(results, output_file)
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_results(results: dict, path: str | Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(path: str | Path) -> dict:
    """Load evaluation results from JSON."""
    with open(path) as f:
        return json.load(f)
