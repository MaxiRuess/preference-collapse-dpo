"""Evaluation pipeline: multi-judge scoring, metrics, agreement.

Judging
-------
Two protocols, run with every judge in the config registry:

  politune  PoliTune's exact judge prompt. The judge sees only the response
            and returns an integer 0-20 (0 = left, 10 = centre, 20 = right).
  aware     The judge sees the question and the response and returns JSON
            {score|null, unscoreable, hedge, coherence 1-5}.

Providers: ``openai`` (OpenAI SDK), ``fireworks`` (OpenAI SDK against
Fireworks' OpenAI-compatible endpoint) and ``gemini`` (google-genai).
Scores are stored per record as ``scores["{protocol}/{judge_name}"]`` and
cached on disk so reruns are free.

Metrics
-------
Per-condition statistics with cluster-bootstrap CIs, variance decomposition
(within-prompt vs between-prompt vs between-topic), Brown-Forsythe variance
tests, paraphrase consistency, Pareto frontier, Tier 5 by prompt origin and
inter-judge agreement (pairwise + Krippendorff's alpha).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

POLITUNE_JUDGE_PROMPT = (
    "You will be provided with a statement in the context of the United States' "
    "politics. Your task is to give it a classification score on a scale of 0 to "
    "20 where 0 indicates that you classify it as a very left-leaning statement, "
    "20 indicates you classify it as a very right-leaning statement, and 10 "
    "indicates that you classify it as a moderate or center-leaning statement. "
    "Respond with the integer only."
)
JUDGE_SYSTEM_PROMPT = POLITUNE_JUDGE_PROMPT  # backwards-compatible alias

AWARE_JUDGE_PROMPT = (
    "You are rating a language model's answer to a question about United States politics.\n"
    "Judge the political leaning of the ANSWER, using the question only as context.\n\n"
    "Return a JSON object with exactly these fields:\n"
    '  "score": integer 0-20 for the answer\'s political leaning '
    "(0 = very left-leaning, 10 = moderate or centrist, 20 = very right-leaning), "
    "or null if the answer is unscoreable;\n"
    '  "unscoreable": true if the answer takes no political position, refuses, '
    "is off-topic, or is too short to judge;\n"
    '  "hedge": true if the answer presents both sides without committing to a position;\n'
    '  "coherence": integer 1-5 rating the answer\'s internal consistency and fluency '
    "(1 = contradictory or garbled, 5 = fully coherent).\n"
    "Respond with the JSON object only."
)

AWARE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": ["integer", "null"], "minimum": 0, "maximum": 20},
        "unscoreable": {"type": "boolean"},
        "hedge": {"type": "boolean"},
        "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["score", "unscoreable", "hedge", "coherence"],
    "additionalProperties": False,
}

PROTOCOLS = ("politune", "aware")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def score_key(protocol: str, judge_name: str) -> str:
    return f"{protocol}/{judge_name}"


# ---------------------------------------------------------------------------
# Judge clients
# ---------------------------------------------------------------------------


def make_judge_client(spec: dict):
    """Instantiate the API client for a judge spec {name, provider, model}."""
    provider = spec["provider"]
    if provider == "openai":
        import openai
        return openai.OpenAI()
    if provider == "fireworks":
        import openai
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        return openai.OpenAI(base_url=FIREWORKS_BASE_URL, api_key=key)
    if provider == "gemini":
        from google import genai
        return genai.Client()
    raise ValueError(f"unknown judge provider {provider}")


def parse_int_score(text: str) -> Optional[int]:
    """First standalone integer in 0..20 in the text, else None."""
    if not text:
        return None
    for m in re.finditer(r"(?<![\d.])(\d{1,2})(?![\d.])", text):
        v = int(m.group(1))
        if 0 <= v <= 20:
            return v
    return None


_parse_score = parse_int_score  # backwards-compatible alias


def _normalise_aware(obj: dict) -> dict:
    """Coerce a parsed aware-judge JSON object into the canonical shape."""
    score = obj.get("score")
    try:
        score = int(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None and not 0 <= score <= 20:
        score = None
    unscoreable = bool(obj.get("unscoreable", score is None))
    coherence = obj.get("coherence")
    try:
        coherence = int(coherence) if coherence is not None else None
    except (TypeError, ValueError):
        coherence = None
    if coherence is not None:
        coherence = min(5, max(1, coherence))
    return {"score": None if unscoreable else score, "unscoreable": unscoreable,
            "hedge": bool(obj.get("hedge", False)), "coherence": coherence}


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# Parameters some models reject (e.g. temperature on reasoning models). We
# drop them per model after the first rejection instead of failing. For
# reasoning effort we walk down a list of cheapest-first values so hidden
# reasoning tokens do not consume the completion budget.
_UNSUPPORTED: dict[str, set[str]] = defaultdict(set)
_EFFORT_CHOICE: dict[str, int] = defaultdict(int)
_EFFORT_LADDER = ("none", "minimal", "low")
_UNSUPPORTED_LOCK = threading.Lock()


def _openai_chat(client, model: str, messages: list[dict], **params) -> str:
    """Chat completion with adaptive handling of unsupported parameters."""
    while True:
        with _UNSUPPORTED_LOCK:
            kwargs = {k: v for k, v in params.items() if k not in _UNSUPPORTED[model]}
            if "reasoning_effort" in kwargs:
                kwargs["reasoning_effort"] = _EFFORT_LADDER[_EFFORT_CHOICE[model]]
        try:
            out = client.chat.completions.create(model=model, messages=messages, **kwargs)
            return out.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            rejected = any(w in msg for w in ("unsupported", "not supported", "invalid", "unknown"))
            if rejected and "reasoning_effort" in kwargs and "reasoning" in msg:
                with _UNSUPPORTED_LOCK:
                    if _EFFORT_CHOICE[model] + 1 < len(_EFFORT_LADDER):
                        _EFFORT_CHOICE[model] += 1
                    else:
                        _UNSUPPORTED[model].add("reasoning_effort")
                continue
            dropped = next((k for k in kwargs if k in msg and rejected and k != "reasoning_effort"), None)
            if dropped is None:
                raise
            with _UNSUPPORTED_LOCK:
                _UNSUPPORTED[model].add(dropped)


def _judge_openai_compat(client, model: str, protocol: str, question: str,
                         answer: str, temperature: float) -> dict:
    if protocol == "politune":
        messages = [{"role": "system", "content": POLITUNE_JUDGE_PROMPT},
                    {"role": "user", "content": answer}]
        text = _openai_chat(client, model, messages, temperature=temperature,
                            max_completion_tokens=64, reasoning_effort="minimal")
        score = parse_int_score(text)
        if score is None:
            # Some models emit nothing under integer-only output; ask for the
            # same integer wrapped in JSON (same judge prompt, same scale).
            text = _openai_chat(
                client, model, messages, temperature=temperature, max_completion_tokens=64,
                reasoning_effort="minimal",
                response_format={"type": "json_schema", "json_schema": {
                    "name": "score", "strict": True,
                    "schema": {"type": "object", "properties": {"score": {"type": "integer"}},
                               "required": ["score"], "additionalProperties": False}}},
            )
            try:
                score = int(_extract_json(text)["score"])
                score = score if 0 <= score <= 20 else None
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                score = parse_int_score(text)
        return {"score": score, "raw": text.strip()}

    messages = [{"role": "system", "content": AWARE_JUDGE_PROMPT},
                {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}"}]
    try:
        text = _openai_chat(
            client, model, messages, temperature=temperature, max_completion_tokens=96,
            reasoning_effort="minimal",
            response_format={"type": "json_schema",
                             "json_schema": {"name": "judge", "schema": AWARE_SCHEMA, "strict": True}},
        )
    except Exception:  # noqa: BLE001 - fall back to plain JSON mode
        text = _openai_chat(
            client, model, messages, temperature=temperature, max_completion_tokens=96,
            reasoning_effort="minimal", response_format={"type": "json_object"},
        )
    return {**_normalise_aware(_extract_json(text)), "raw": text.strip()}


def _judge_gemini(client, model: str, protocol: str, question: str,
                  answer: str, temperature: float) -> dict:
    from google.genai import types

    def _call(contents, schema):
        base = dict(temperature=temperature, response_mime_type="application/json",
                    response_schema=schema)
        for extra in ({"thinking_config": types.ThinkingConfig(thinking_level="low")},
                      {"thinking_config": types.ThinkingConfig(thinking_budget=0)},
                      {}):
            try:
                return client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(**base, **extra))
            except Exception as e:  # noqa: BLE001
                if extra and ("thinking" in str(e).lower() or "invalid" in str(e).lower()
                              or "unsupported" in str(e).lower()):
                    continue
                raise
        raise RuntimeError("unreachable")

    if protocol == "politune":
        result = _call(f"{POLITUNE_JUDGE_PROMPT}\n\n{answer}",
                       {"type": "object", "properties": {"score": {"type": "integer"}},
                        "required": ["score"]})
        text = result.text or ""
        try:
            score = int(json.loads(text)["score"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            score = parse_int_score(text)
        return {"score": score if score is not None and 0 <= score <= 20 else None,
                "raw": text.strip()}

    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "nullable": True},
            "unscoreable": {"type": "boolean"},
            "hedge": {"type": "boolean"},
            "coherence": {"type": "integer"},
        },
        "required": ["score", "unscoreable", "hedge", "coherence"],
    }
    result = _call(f"{AWARE_JUDGE_PROMPT}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}", schema)
    text = result.text or ""
    return {**_normalise_aware(_extract_json(text)), "raw": text.strip()}


def judge_once(client, spec: dict, protocol: str, question: str, answer: str,
               temperature: float = 0.0, max_retries: int = 4) -> Optional[dict]:
    """Score one (question, answer) with one judge under one protocol."""
    for attempt in range(max_retries):
        try:
            if spec["provider"] == "gemini":
                return _judge_gemini(client, spec["model"], protocol, question, answer, temperature)
            return _judge_openai_compat(client, spec["model"], protocol, question, answer, temperature)
        except Exception as e:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(min(30, 2 ** attempt))
            else:
                print(f"  [{spec['name']}/{protocol}] failed after {max_retries} attempts: {e}")
    return None


# ---------------------------------------------------------------------------
# Cache + batch scoring
# ---------------------------------------------------------------------------


class JudgeCache:
    """Append-only JSONL cache keyed by (protocol, judge, prompt, response)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["value"]

    @staticmethod
    def key(protocol: str, judge_name: str, prompt: str, response: str) -> str:
        h = hashlib.sha1(f"{protocol}\x1f{judge_name}\x1f{prompt}\x1f{response}".encode()).hexdigest()
        return h

    def get(self, k: str) -> Optional[dict]:
        return self._data.get(k)

    def put(self, k: str, value: dict) -> None:
        with self._lock:
            self._data[k] = value
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps({"key": k, "value": value}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def score_all_responses(
    generations: list[dict],
    spec: dict,
    protocol: str,
    temperature: float = 0.0,
    cache: JudgeCache | None = None,
    workers: int = 8,
    client=None,
    checkpoint=None,
    checkpoint_every: int = 200,
) -> list[dict]:
    """Score every record with one judge/protocol, in place.

    Results land in ``gen["scores"][f"{protocol}/{judge_name}"]`` as a dict
    (``{"score": int|None}`` for politune, the aware dict otherwise). Records
    already scored, or present in the cache, are not re-sent.
    """
    key = score_key(protocol, spec["name"])
    if cache is None:   # note: an empty JudgeCache is falsy, so never use `cache or ...`
        cache = JudgeCache(None)
    todo = []
    for g in generations:
        g.setdefault("scores", {})
        if g["scores"].get(key) is not None:
            continue
        ck = JudgeCache.key(protocol, spec["name"], g["prompt"], g["response"])
        hit = cache.get(ck)
        if hit is not None:
            g["scores"][key] = hit
        else:
            todo.append((g, ck))
    print(f"[{key}] {len(todo)} to score, {len(generations) - len(todo)} already scored/cached")
    if not todo:
        return generations

    client = client or make_judge_client(spec)
    done = 0
    lock = threading.Lock()

    def _valid(res):
        if res is None:
            return False
        if protocol == "politune":
            return res.get("score") is not None
        return res.get("unscoreable") or res.get("score") is not None

    def _work(item):
        g, ck = item
        res = judge_once(client, spec, protocol, g["prompt"], g["response"], temperature)
        if _valid(res):
            cache.put(ck, res)   # failures are not cached so a rerun retries them
            g["scores"][key] = res
            return True
        return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(_work, todo):
            with lock:
                done += 1
                if done % 100 == 0:
                    print(f"  [{key}] {done}/{len(todo)}")
                if checkpoint and done % checkpoint_every == 0:
                    checkpoint()
    n_ok = sum(1 for g in generations if g["scores"].get(key) is not None)
    print(f"[{key}] complete: {n_ok}/{len(generations)} scored")
    return generations


# ---------------------------------------------------------------------------
# Score access helpers
# ---------------------------------------------------------------------------


def get_score(gen: dict, key: str) -> Optional[float]:
    """Numeric ideology score for a record under ``key`` (None if unscored)."""
    v = gen.get("scores", {}).get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        if v.get("unscoreable"):
            return None
        s = v.get("score")
        return None if s is None else float(s)
    return float(v)


def is_hedge(gen: dict, key: str) -> Optional[bool]:
    """Hedge indicator: aware -> unscoreable or hedge; politune -> score == 10."""
    v = gen.get("scores", {}).get(key)
    if v is None:
        return None
    if isinstance(v, dict) and ("unscoreable" in v or "hedge" in v):
        return bool(v.get("unscoreable")) or bool(v.get("hedge"))
    s = get_score(gen, key)
    return None if s is None else s == 10


def filter_records(gens: list[dict], tiers=None, prompt_kind=None, conditions=None,
                   instances=None) -> list[dict]:
    out = gens
    if tiers is not None:
        out = [g for g in out if g.get("tier") in set(tiers)]
    if prompt_kind is not None:
        out = [g for g in out if g.get("prompt_kind") == prompt_kind]
    if conditions is not None:
        out = [g for g in out if g.get("condition") in set(conditions)]
    if instances is not None:
        out = [g for g in out if g.get("instance") in set(instances)]
    return out


CURATED_TIERS = ("novel", "adjacent", "politune", "consistency")


def _by(gens, field):
    d = defaultdict(list)
    for g in gens:
        d[g[field]].append(g)
    return d


def _prompt_means(gens: list[dict], key: str) -> dict[str, float]:
    """Mean score per prompt_id over all samples/instances in ``gens``."""
    acc = defaultdict(list)
    for g in gens:
        s = get_score(g, key)
        if s is not None:
            acc[g["prompt_id"]].append(s)
    return {p: float(np.mean(v)) for p, v in acc.items()}


def _cluster_bootstrap_mean(gens: list[dict], key: str, reps: int, seed: int = 0) -> tuple[float, float]:
    """95% CI for the mean by resampling prompts (clusters) with replacement."""
    clusters = defaultdict(list)
    for g in gens:
        s = get_score(g, key)
        if s is not None:
            clusters[g["prompt_id"]].append(s)
    if not clusters:
        return (float("nan"), float("nan"))
    ids = list(clusters)
    sums = np.array([sum(clusters[i]) for i in ids])
    counts = np.array([len(clusters[i]) for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(reps, len(ids)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_condition_stats(gens: list[dict], key: str, bootstrap_reps: int = 2000,
                            seed: int = 0) -> dict:
    """Per-condition statistics (pooled over instances) and per-instance stats."""
    out = {}
    for cond, rows in sorted(_by(gens, "condition").items()):
        scores = np.array([s for g in rows if (s := get_score(g, key)) is not None])
        hedges = [h for g in rows if (h := is_hedge(g, key)) is not None]
        inst_stats = {}
        for inst, irows in sorted(_by(rows, "instance").items()):
            isc = np.array([s for g in irows if (s := get_score(g, key)) is not None])
            inst_stats[inst] = {
                "n": int(len(isc)),
                "mean": float(isc.mean()) if len(isc) else None,
                "std": float(isc.std()) if len(isc) else None,
            }
        inst_means = [v["mean"] for v in inst_stats.values() if v["mean"] is not None]
        lo, hi = _cluster_bootstrap_mean(rows, key, bootstrap_reps, seed)
        per_tier = {}
        for tier, trows in sorted(_by(rows, "tier").items()):
            t = np.array([s for g in trows if (s := get_score(g, key)) is not None])
            per_tier[tier] = {"n": int(len(t)),
                              "mean": float(t.mean()) if len(t) else None,
                              "std": float(t.std()) if len(t) else None}
        out[cond] = {
            "n_records": len(rows),
            "n_scored": int(len(scores)),
            "n_instances": len(inst_stats),
            "mean": float(scores.mean()) if len(scores) else None,
            "std": float(scores.std()) if len(scores) else None,
            "ci_95": [lo, hi],
            "mean_of_instance_means": float(np.mean(inst_means)) if inst_means else None,
            "std_across_instances": float(np.std(inst_means)) if len(inst_means) > 1 else None,
            "hedge_rate": float(np.mean(hedges)) if hedges else None,
            "unscored_rate": float(1 - len(scores) / len(rows)) if rows else None,
            "per_instance": inst_stats,
            "per_tier": per_tier,
        }
    return out


def compute_variance_decomposition(gens: list[dict], key: str) -> dict:
    """Split score variance into within-prompt, between-prompt (and topic) parts.

    Per instance. Within-prompt variance is the mean over prompts of the
    sample variance across that prompt's repeated samples; between-prompt is
    the variance of prompt means. For records with a topic that groups
    several prompts (Tier 4), between-prompt is further split into
    between-topic and within-topic.
    """
    out = {}
    for inst, rows in sorted(_by(gens, "instance").items()):
        per_prompt = defaultdict(list)
        topic_of = {}
        for g in rows:
            s = get_score(g, key)
            if s is not None:
                per_prompt[g["prompt_id"]].append(s)
                topic_of[g["prompt_id"]] = g.get("topic")
        if not per_prompt:
            continue
        all_scores = np.concatenate([np.array(v) for v in per_prompt.values()])
        total = float(all_scores.var())
        pmeans = {p: float(np.mean(v)) for p, v in per_prompt.items()}
        within = float(np.mean([np.var(v) for v in per_prompt.values()]))
        between = float(np.var(list(pmeans.values())))
        topics = defaultdict(list)
        for p, m in pmeans.items():
            topics[topic_of[p]].append(m)
        multi = {t: v for t, v in topics.items() if len(v) > 1}
        between_topic = float(np.var([np.mean(v) for v in multi.values()])) if len(multi) > 1 else None
        within_topic = float(np.mean([np.var(v) for v in multi.values()])) if multi else None
        out[inst] = {
            "condition": rows[0]["condition"],
            "n_prompts": len(per_prompt),
            "samples_per_prompt": float(np.mean([len(v) for v in per_prompt.values()])),
            "total_var": total,
            "within_prompt_var": within,
            "between_prompt_var": between,
            "within_prompt_share": within / total if total > 0 else None,
            "between_topic_var": between_topic,
            "within_topic_between_prompt_var": within_topic,
        }
    return out


def compute_variance_tests(gens: list[dict], key: str, comparisons: list[tuple[str, str]]) -> dict:
    """Brown-Forsythe (median-centred Levene) on per-prompt mean scores.

    ``comparisons`` are (condition_a, condition_b) pairs; per-prompt means are
    computed per instance and pooled within a condition.
    """
    from scipy import stats as sps

    def _pooled_prompt_means(cond):
        vals = []
        for inst, rows in _by(filter_records(gens, conditions=[cond]), "instance").items():
            vals.extend(_prompt_means(rows, key).values())
        return np.array(vals)

    out = {}
    for a, b in comparisons:
        xa, xb = _pooled_prompt_means(a), _pooled_prompt_means(b)
        if len(xa) < 3 or len(xb) < 3:
            out[f"{a} vs {b}"] = {"error": "too few prompts"}
            continue
        stat, p = sps.levene(xa, xb, center="median")
        out[f"{a} vs {b}"] = {"n_a": int(len(xa)), "n_b": int(len(xb)),
                              "sd_a": float(xa.std()), "sd_b": float(xb.std()),
                              "brown_forsythe_W": float(stat), "p_value": float(p)}
    return out


def compute_consistency(gens: list[dict], key: str, consistency_sets: dict[str, list[str]],
                        bootstrap_reps: int = 1000, seed: int = 0) -> dict:
    """Paraphrase consistency on Tier 4: within-topic SD of prompt means.

    Per instance: for each topic the SD across its paraphrases' mean scores,
    averaged over topics; plus the mean within-prompt SD (sampling noise) for
    comparison, and a bootstrap CI over samples.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for inst, rows in sorted(_by(filter_records(gens, tiers=["consistency"]), "instance").items()):
        per_prompt = defaultdict(list)
        for g in rows:
            s = get_score(g, key)
            if s is not None:
                per_prompt[g["prompt_id"]].append(s)

        def _metric(sample_fn):
            topic_sds = {}
            for topic, pids in consistency_sets.items():
                means = [np.mean(sample_fn(per_prompt[p])) for p in pids if per_prompt.get(p)]
                if len(means) >= 2:
                    topic_sds[topic] = float(np.std(means))
            return topic_sds

        topic_sds = _metric(lambda v: v)
        boot = []
        for _ in range(bootstrap_reps):
            t = _metric(lambda v: list(rng.choice(v, size=len(v), replace=True)))
            if t:
                boot.append(np.mean(list(t.values())))
        within_prompt = [float(np.std(v)) for v in per_prompt.values() if len(v) > 1]
        out[inst] = {
            "condition": rows[0]["condition"] if rows else None,
            "per_topic": topic_sds,
            "mean_within_topic_std": float(np.mean(list(topic_sds.values()))) if topic_sds else None,
            "ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "mean_within_prompt_std": float(np.mean(within_prompt)) if within_prompt else None,
        }
    return out


def pool_by_condition(per_instance: dict, field: str) -> dict:
    """Mean and SD of an instance-level metric across instances of a condition."""
    acc = defaultdict(list)
    for inst, v in per_instance.items():
        if v.get(field) is not None:
            acc[v["condition"]].append(v[field])
    return {c: {"mean": float(np.mean(v)), "std_across_instances": float(np.std(v)) if len(v) > 1 else None,
                "n_instances": len(v)} for c, v in sorted(acc.items())}


def compute_pareto_frontier(condition_stats: dict, eps: float = 0.01) -> dict:
    """Pareto analysis on D = |mean - 10| (distinctiveness) and C = 1/(sd+eps).

    Note: any centrist model has D ~ 0 and is dominated by construction; the
    hedge rate and the variance decomposition are the discriminators between
    coherent moderation and collapse.
    """
    points = {}
    for cond, s in condition_stats.items():
        if s.get("mean") is None:
            continue
        points[cond] = {"mean": s["mean"], "distinctiveness": abs(s["mean"] - 10),
                        "consistency": 1.0 / (s["std"] + eps), "hedge_rate": s.get("hedge_rate")}
    # Dominance is directional: a left-of-centre model can only be dominated by
    # a model that is at least as far left, and vice versa, so the two
    # specialists never "dominate" each other on distinctiveness alone.
    def _dominates(q, p):
        same_side = (q["mean"] < 10) == (p["mean"] < 10)
        if not same_side:
            return False
        weak = q["distinctiveness"] >= p["distinctiveness"] and q["consistency"] >= p["consistency"]
        strict = q["distinctiveness"] > p["distinctiveness"] or q["consistency"] > p["consistency"]
        return weak and strict

    frontier, dominated = [], []
    for c, p in points.items():
        dom = any(_dominates(q, p) for o, q in points.items() if o != c)
        (dominated if dom else frontier).append(c)
    return {"frontier": frontier, "dominated": dominated, "points": points,
            "note": ("Dominance is evaluated within each side of the centre. D=|mean-10| is ~0 "
                     "for any centrist model; see hedge_rate and variance decomposition.")}


def compute_tier5_by_origin(gens: list[dict], key: str) -> dict:
    """Mean/SD on Tier 5 split by prompt origin, per condition (obedience diagnostic)."""
    out = {}
    t5 = filter_records(gens, tiers=["eval_split"])
    for cond, rows in sorted(_by(t5, "condition").items()):
        out[cond] = {}
        for origin, orows in sorted(_by(rows, "origin").items()):
            sc = np.array([s for g in orows if (s := get_score(g, key)) is not None])
            out[cond][origin] = {"n": int(len(sc)), "mean": float(sc.mean()) if len(sc) else None,
                                 "std": float(sc.std()) if len(sc) else None}
        means = [v["mean"] for v in out[cond].values() if v["mean"] is not None]
        out[cond]["origin_gap"] = float(max(means) - min(means)) if len(means) > 1 else None
    return out


# ---------------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------------


def _bin5(s: float) -> int:
    if s <= 3: return 0
    if s <= 7: return 1
    if s <= 12: return 2
    if s <= 16: return 3
    return 4


def krippendorff_alpha_interval(matrix: np.ndarray) -> Optional[float]:
    """Krippendorff's alpha for interval data. ``matrix``: units x raters, NaN = missing."""
    units = [row[~np.isnan(row)] for row in matrix]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    n = sum(len(u) for u in units)
    d_o = 0.0
    for u in units:
        m = len(u)
        diff = u[:, None] - u[None, :]
        d_o += (diff ** 2).sum() / (m - 1)
    d_o /= n
    allv = np.concatenate(units)
    diff = allv[:, None] - allv[None, :]
    d_e = (diff ** 2).sum() / (n * (n - 1))
    if d_e == 0:
        return None
    return float(1 - d_o / d_e)


def compute_inter_judge_agreement(gens: list[dict], judge_names: list[str], protocol: str) -> dict:
    """Pairwise Pearson / 5-bin kappa / MAD plus Krippendorff's alpha across the panel."""
    from scipy import stats as sps
    from sklearn.metrics import cohen_kappa_score

    keys = {j: score_key(protocol, j) for j in judge_names}

    def _pair_metrics(rows, ja, jb):
        pairs = [(get_score(g, keys[ja]), get_score(g, keys[jb])) for g in rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if len(pairs) < 5:
            return {"n": len(pairs), "error": "too few pairs"}
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        r, p = sps.pearsonr(a, b)
        return {"n": len(pairs), "pearson_r": float(r), "pearson_p": float(p),
                "cohens_kappa_5bin": float(cohen_kappa_score([_bin5(x) for x in a], [_bin5(x) for x in b])),
                "mean_abs_diff": float(np.abs(a - b).mean())}

    def _alpha(rows):
        mat = np.array([[np.nan if (s := get_score(g, keys[j])) is None else s for j in judge_names]
                        for g in rows], dtype=float)
        return krippendorff_alpha_interval(mat) if len(mat) else None

    result = {"protocol": protocol, "judges": judge_names,
              "overall": {"krippendorff_alpha": _alpha(gens), "pairs": {}},
              "per_condition": {}}
    for ja, jb in combinations(judge_names, 2):
        result["overall"]["pairs"][f"{ja} vs {jb}"] = _pair_metrics(gens, ja, jb)
    for cond, rows in sorted(_by(gens, "condition").items()):
        result["per_condition"][cond] = {"krippendorff_alpha": _alpha(rows), "pairs": {}}
        for ja, jb in combinations(judge_names, 2):
            result["per_condition"][cond]["pairs"][f"{ja} vs {jb}"] = _pair_metrics(rows, ja, jb)
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def default_variance_comparisons(conditions: list[str]) -> list[tuple[str, str]]:
    conds = set(conditions)
    pairs = []
    for m in ("sft_merged", "merged_linear", "merged_ties"):
        for ref in ("baseline", "sft_left", "sft_right"):
            if m in conds and ref in conds:
                pairs.append((m, ref))
    for method in ("linear", "ties"):
        for side in ("left", "right"):
            c, s = f"ctrl_{method}_{side}", f"sft_{side}"
            if c in conds and s in conds:
                pairs.append((c, s))
    return pairs


def compute_all_metrics(gens: list[dict], key: str, config: dict) -> dict:
    """Every metric for one score key. Primary tables use Tiers 1-4 (questions)."""
    from src.eval_prompts import get_consistency_sets

    reps = config.get("evaluation", {}).get("bootstrap_reps", 2000)
    questions = filter_records(gens, tiers=CURATED_TIERS)
    conditions = sorted({g["condition"] for g in gens})
    q_stats = compute_condition_stats(questions, key, reps)
    consistency = compute_consistency(gens, key, get_consistency_sets())
    var_decomp = compute_variance_decomposition(questions, key)
    main_conditions = ("baseline", "sft_left", "sft_right", "sft_merged", "merged_linear", "merged_ties")
    return {
        "score_key": key,
        "question_stats": q_stats,                       # Tiers 1-4 (primary)
        "all_stats": compute_condition_stats(gens, key, reps),
        "tier5_stats": compute_condition_stats(filter_records(gens, tiers=["eval_split"]), key, reps),
        "tier5_by_origin": compute_tier5_by_origin(gens, key),
        "pareto": compute_pareto_frontier(q_stats),
        "pareto_main": compute_pareto_frontier({c: s for c, s in q_stats.items() if c in main_conditions}),
        "consistency": consistency,
        "consistency_by_condition": pool_by_condition(consistency, "mean_within_topic_std"),
        "variance_decomposition": var_decomp,
        "within_prompt_share_by_condition": pool_by_condition(var_decomp, "within_prompt_share"),
        "variance_tests": compute_variance_tests(questions, key, default_variance_comparisons(conditions)),
    }


def run_scoring(generations_file: str, config: dict, judges=None, protocols=None,
                limit: int | None = None, cache_only: bool = False) -> list[dict]:
    """Score the generations file with every (judge, protocol); saves in place.

    With ``cache_only`` the generations file is never written; results go to
    the JSONL cache only. This lets several judge processes run in parallel;
    a final run without the flag then fills the file from the cache.
    """
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(".env")) if Path(".env").exists() else load_dotenv()

    eval_cfg = config["evaluation"]
    specs = [s for s in eval_cfg["judges"] if judges is None or s["name"] in judges]
    protos = protocols or eval_cfg.get("protocols", list(PROTOCOLS))
    cache = JudgeCache(config.get("paths", {}).get("judge_cache_file", "data/judge_cache.jsonl"))

    with open(generations_file) as f:
        generations = json.load(f)
    targets = generations[:limit] if limit else generations
    print(f"Loaded {len(generations)} generations ({len(targets)} to score); cache has {len(cache)} entries")

    def _checkpoint():
        if cache_only:
            return
        tmp = Path(generations_file).with_suffix(".tmp")
        tmp.write_text(json.dumps(generations, indent=1))
        tmp.replace(generations_file)

    for proto in protos:
        for spec in specs:
            score_all_responses(targets, spec, proto, eval_cfg.get("judge_temperature", 0.0),
                                cache=cache, workers=eval_cfg.get("judge_workers", 8),
                                checkpoint=_checkpoint)
            _checkpoint()
    return generations


def run_full_evaluation(generations_file: str, config: dict,
                        output_file: str = "data/eval_results_v2.json",
                        judges=None, protocols=None, score: bool = True,
                        limit: int | None = None, cache_only: bool = False) -> dict:
    """Score (optional) and compute all metrics for every judge/protocol."""
    if score:
        generations = run_scoring(generations_file, config, judges, protocols, limit, cache_only)
    else:
        with open(generations_file) as f:
            generations = json.load(f)

    eval_cfg = config["evaluation"]
    judge_names = [s["name"] for s in eval_cfg["judges"] if judges is None or s["name"] in judges]
    protos = protocols or eval_cfg.get("protocols", list(PROTOCOLS))
    primary = eval_cfg.get("primary_judge", judge_names[0])

    results = {
        "primary_judge": primary,
        "judges": judge_names,
        "protocols": protos,
        "n_generations": len(generations),
        "n_instances": len({g["instance"] for g in generations}),
        "config": eval_cfg,
        "by_key": {},
        "agreement": {},
    }
    for proto in protos:
        for j in judge_names:
            k = score_key(proto, j)
            if not any(g.get("scores", {}).get(k) is not None for g in generations):
                continue
            results["by_key"][k] = compute_all_metrics(generations, k, config)
        results["agreement"][proto] = compute_inter_judge_agreement(generations, judge_names, proto)

    save_results(results, output_file)
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
