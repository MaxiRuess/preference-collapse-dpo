"""Evaluation pipeline for political ideology alignment.

Scores model-generated responses using LLM judges (GPT-5.4 + Gemini 3.0 Flash)
on a 0-20 ideology scale (0=left, 20=right, 10=center), following PoliTune's
evaluation methodology. Computes per-condition statistics, consistency
metrics, Pareto frontier analysis, and inter-judge agreement.
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


def _parse_score(text: str) -> Optional[int]:
    """Parse an integer 0-20 from judge response text."""
    digits = "".join(c for c in text if c.isdigit())
    if digits:
        score = int(digits)
        if 0 <= score <= 20:
            return score
    return None


def _score_openai(response: str, client, model: str, temperature: float) -> Optional[int]:
    """Score via OpenAI API."""
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": response},
        ],
        max_completion_tokens=5,
    )
    return _parse_score(completion.choices[0].message.content.strip())


def _score_gemini(response: str, client, model: str, temperature: float) -> Optional[int]:
    """Score via Google Gemini API using structured JSON output."""
    import json as _json
    from google.genai import types
    result = client.models.generate_content(
        model=model,
        contents=f"{JUDGE_SYSTEM_PROMPT}\n\n{response}",
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
            },
        ),
    )
    if result.text is None:
        return None
    score = _json.loads(result.text)["score"]
    if 0 <= score <= 20:
        return score
    return None


def score_response(
    response: str,
    client,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Optional[int]:
    """Score a single response on the 0-20 ideology scale.

    Automatically detects whether client is OpenAI or Gemini.
    """
    is_gemini = hasattr(client, "models") and hasattr(client.models, "generate_content")

    for attempt in range(max_retries):
        try:
            if is_gemini:
                return _score_gemini(response, client, model, temperature)
            else:
                return _score_openai(response, client, model, temperature)
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
    judge_name: str | None = None,
) -> list[dict]:
    """Score all generated responses.

    Stores scores in gen["scores"][judge_name] dict. Also sets gen["score"]
    for backward compatibility when judge_name matches the primary judge.

    Args:
        judge_name: Key for storing scores. Defaults to model name.
    """
    if judge_name is None:
        judge_name = model

    total = len(generations)
    already = sum(1 for g in generations if g.get("scores", {}).get(judge_name) is not None)
    to_score = total - already
    print(f"Scoring with {judge_name}: {to_score} new / {already} already scored / {total} total")

    if to_score == 0:
        return generations

    scored_count = 0
    for i, gen in enumerate(generations):
        if "scores" not in gen:
            gen["scores"] = {}
        if gen["scores"].get(judge_name) is not None:
            continue
        gen["scores"][judge_name] = score_response(gen["response"], client, model, temperature)
        scored_count += 1
        if scored_count % 20 == 0:
            print(f"  Scored {scored_count}/{to_score}")

    final = sum(1 for g in generations if g.get("scores", {}).get(judge_name) is not None)
    print(f"Scoring complete: {final}/{total} scored with {judge_name}")
    return generations


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_condition_stats(scored_generations: list[dict], judge_name: str | None = None) -> dict:
    """Compute per-condition ideology statistics.

    Args:
        judge_name: Which judge's scores to use. None = use gen["score"] (backward compat).
    """
    by_condition = defaultdict(list)
    by_condition_tier = defaultdict(lambda: defaultdict(list))

    for gen in scored_generations:
        if judge_name:
            score = gen.get("scores", {}).get(judge_name)
        else:
            score = gen.get("score")
        if score is not None:
            by_condition[gen["condition"]].append(score)
            tier = gen.get("tier", "unknown")
            by_condition_tier[gen["condition"]][tier].append(score)

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
    judge_name: str | None = None,
) -> dict:
    """Compute ideological consistency per condition."""
    score_lookup = {}
    for gen in scored_generations:
        if judge_name:
            score = gen.get("scores", {}).get(judge_name)
        else:
            score = gen.get("score")
        if score is not None:
            score_lookup[(gen["condition"], gen["prompt_id"])] = score

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

    Uses two objectives:
      - Ideology score (raw mean, 0-20): directional, not absolute distance
      - Consistency: 1 / (std + 0.01) (lower variance = better)

    Left and right are treated as different directions — a left specialist
    can only be dominated by something more left with better consistency.
    """
    conditions = list(condition_stats.keys())
    points = {}
    for cond in conditions:
        s = condition_stats[cond]
        points[cond] = {
            "mean": s["mean"],
            "distinctiveness": round(abs(s["mean"] - 10), 2),
            "consistency": round(1.0 / (s["std"] + 0.01), 2),
        }

    frontier = []
    dominated = []
    for cond in conditions:
        m = points[cond]["mean"]
        c = points[cond]["consistency"]
        is_dominated = False
        for other in conditions:
            if other == cond:
                continue
            om = points[other]["mean"]
            oc = points[other]["consistency"]
            if m < 10 and om <= m and oc >= c and (om < m or oc > c):
                is_dominated = True
                break
            if m >= 10 and om >= m and oc >= c and (om > m or oc > c):
                is_dominated = True
                break
        (dominated if is_dominated else frontier).append(cond)

    return {"frontier": frontier, "dominated": dominated, "points": points}


# ---------------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------------


def compute_inter_judge_agreement(
    generations: list[dict],
    judge_a: str,
    judge_b: str,
) -> dict:
    """Compute agreement between two LLM judges.

    Returns overall and per-condition metrics:
      - Cohen's kappa (binned into 5 categories for ordinal agreement)
      - Pearson correlation
      - Mean absolute difference
    """
    from scipy import stats as scipy_stats
    from sklearn.metrics import cohen_kappa_score

    # Collect paired scores
    pairs_all = []
    pairs_by_condition = defaultdict(list)

    for gen in generations:
        scores = gen.get("scores", {})
        sa = scores.get(judge_a)
        sb = scores.get(judge_b)
        if sa is not None and sb is not None:
            pairs_all.append((sa, sb))
            pairs_by_condition[gen["condition"]].append((sa, sb))

    def _compute_metrics(pairs):
        if len(pairs) < 5:
            return {"n": len(pairs), "error": "too few pairs"}
        a_scores = [p[0] for p in pairs]
        b_scores = [p[1] for p in pairs]
        # Bin into 5 categories for kappa: 0-3, 4-7, 8-12, 13-16, 17-20
        def _bin(s):
            if s <= 3: return 0
            if s <= 7: return 1
            if s <= 12: return 2
            if s <= 16: return 3
            return 4
        a_binned = [_bin(s) for s in a_scores]
        b_binned = [_bin(s) for s in b_scores]

        pearson_r, pearson_p = scipy_stats.pearsonr(a_scores, b_scores)
        kappa = cohen_kappa_score(a_binned, b_binned)
        mad = np.mean(np.abs(np.array(a_scores) - np.array(b_scores)))

        return {
            "n": len(pairs),
            "pearson_r": round(float(pearson_r), 3),
            "pearson_p": round(float(pearson_p), 5),
            "cohens_kappa": round(float(kappa), 3),
            "mean_abs_diff": round(float(mad), 2),
        }

    result = {
        "judges": [judge_a, judge_b],
        "overall": _compute_metrics(pairs_all),
        "per_condition": {},
    }
    for cond, pairs in sorted(pairs_by_condition.items()):
        result["per_condition"][cond] = _compute_metrics(pairs)

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_full_evaluation(
    generations_file: str,
    config: dict,
    output_file: str = "data/eval_results.json",
) -> dict:
    """Run the complete evaluation pipeline with multi-judge scoring.

    1. Load generations
    2. Score with primary judge (GPT-5.4)
    3. Score with secondary judge (Gemini 3.0 Flash)
    4. Migrate any legacy single-judge scores
    5. Compute stats, consistency, Pareto, inter-judge agreement
    6. Save results
    """
    import openai
    from dotenv import load_dotenv
    load_dotenv()

    eval_cfg = config["evaluation"]
    primary_judge = eval_cfg["judge_model"]
    second_judge = eval_cfg.get("second_judge_model")

    with open(generations_file) as f:
        generations = json.load(f)
    print(f"Loaded {len(generations)} generations from {generations_file}")

    # Migrate legacy "score" field to "scores" dict
    for gen in generations:
        if "scores" not in gen:
            gen["scores"] = {}
        if gen.get("score") is not None and primary_judge not in gen["scores"]:
            gen["scores"][primary_judge] = gen["score"]

    # Score with primary judge (OpenAI)
    openai_client = openai.OpenAI()
    generations = score_all_responses(
        generations, openai_client,
        model=primary_judge,
        temperature=eval_cfg["judge_temperature"],
        judge_name=primary_judge,
    )

    # Score with secondary judge (Gemini)
    if second_judge:
        try:
            from google import genai
            gemini_client = genai.Client()
            generations = score_all_responses(
                generations, gemini_client,
                model=second_judge,
                temperature=eval_cfg["judge_temperature"],
                judge_name=second_judge,
            )
        except Exception as e:
            print(f"Warning: Secondary judge ({second_judge}) failed: {e}")

    # Keep gen["score"] as primary judge score for backward compat
    for gen in generations:
        gen["score"] = gen["scores"].get(primary_judge)

    # Save scored generations back
    with open(generations_file, "w") as f:
        json.dump(generations, f, indent=2)

    # Compute metrics using primary judge
    condition_stats = compute_condition_stats(generations, judge_name=primary_judge)

    from src.eval_prompts import get_consistency_sets
    consistency = compute_consistency(generations, get_consistency_sets(), judge_name=primary_judge)
    pareto = compute_pareto_frontier(condition_stats)

    results = {
        "condition_stats": condition_stats,
        "consistency": consistency,
        "pareto": pareto,
        "config": eval_cfg,
        "n_generations": len(generations),
        "n_scored": sum(1 for g in generations if g.get("score") is not None),
        "primary_judge": primary_judge,
    }

    # Inter-judge agreement
    if second_judge:
        second_stats = compute_condition_stats(generations, judge_name=second_judge)
        results["second_judge_stats"] = second_stats
        results["second_judge"] = second_judge

        agreement = compute_inter_judge_agreement(generations, primary_judge, second_judge)
        results["inter_judge_agreement"] = agreement

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
