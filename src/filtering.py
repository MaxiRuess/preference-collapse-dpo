"""LLM-as-judge quality and contrast filtering.

Step 3 of the pipeline. For each prompt, forms 2 ideological pairs
(holding epistemic style constant) and judges each on:
  - Contrast (1–5): how clearly do responses represent opposite ideological poles?
  - Quality A (1–5): is the optimist response well-argued and substantive?
  - Quality B (1–5): is the skeptic response well-argued and substantive?

A prompt is kept only if ALL its pairs pass the thresholds.
Uses async OpenAI client with checkpoint resume for crash resilience.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm


# Filter on confident pairs only — these have clear ideological contrast.
# Uncertain pairs are kept for analysis but don't gate filtering.
PAIR_SPECS = [
    ("optimist_confident", "skeptic_confident"),
]

JUDGE_SYSTEM_PROMPT = """\
You are an expert judge evaluating two responses to a question about AI's \
economic impact. Response A comes from a techno-optimist perspective and \
Response B comes from a techno-skeptic perspective.

Score the pair on three dimensions:

1. **contrast** (1-5): How clearly do the two responses represent genuinely \
   different ideological positions?
   - 1: Both say essentially the same thing
   - 3: Some difference but overlap significantly
   - 5: Clearly opposite positions with distinct arguments and conclusions

2. **quality_a** (1-5): How well-argued is Response A (the optimist response)?
   - 1: Shallow, vague, or incoherent
   - 3: Makes some points but lacks depth or evidence
   - 5: Substantive, well-reasoned, uses specific arguments and evidence

3. **quality_b** (1-5): How well-argued is Response B (the skeptic response)?
   - Same scale as quality_a

Respond ONLY with JSON, no other text:
{"contrast": <int>, "quality_a": <int>, "quality_b": <int>, "reasoning": "<brief explanation>"}\
"""


def _get_client(config: dict) -> AsyncOpenAI:
    """Create an async OpenAI client from config."""
    return AsyncOpenAI(api_key=config["openai_api_key"])


def _load_scored_ids(path: Path) -> dict[str, dict]:
    """Load already-scored pair IDs from checkpoint file.

    Returns dict mapping "promptid_pairindex" -> score dict.
    """
    scored = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    key = f"{rec['prompt_id']}_{rec['pair_index']}"
                    scored[key] = rec
    return scored


async def _judge_one(
    client: AsyncOpenAI,
    prompt: str,
    response_a: str,
    response_b: str,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge a single pair using the LLM judge."""
    filter_cfg = config["filtering"]

    user_message = (
        f"**Question:** {prompt}\n\n"
        f"**Response A (optimist):**\n{response_a}\n\n"
        f"**Response B (skeptic):**\n{response_b}"
    )

    async with semaphore:
        response = await client.chat.completions.create(
            model=filter_cfg["judge_model"],
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=filter_cfg["judge_temperature"],
            max_completion_tokens=256,
        )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        parsed = json.loads(raw)
        return {
            "contrast": int(parsed["contrast"]),
            "quality_a": int(parsed["quality_a"]),
            "quality_b": int(parsed["quality_b"]),
            "reasoning": parsed.get("reasoning", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # If parsing fails, return failing scores so the pair gets dropped
        return {
            "contrast": 0,
            "quality_a": 0,
            "quality_b": 0,
            "reasoning": f"PARSE_ERROR: {raw[:200]}",
        }


def judge_pair(
    response_a: str,
    response_b: str,
    prompt: str,
    axis: str,
    config: dict,
) -> dict:
    """Score a single response pair using an LLM judge (sync wrapper).

    Args:
        response_a: First response text (optimist).
        response_b: Second response text (skeptic).
        prompt: The original prompt.
        axis: Which axis is being compared ("ideological" or "epistemic").
        config: Configuration dict.

    Returns:
        Dict with keys: contrast (int 1-5), quality_a (int 1-5),
        quality_b (int 1-5), reasoning (str).
    """
    client = _get_client(config)
    semaphore = asyncio.Semaphore(1)
    return asyncio.run(
        _judge_one(client, prompt, response_a, response_b, config, semaphore)
    )


async def _filter_responses_async(
    responses: list[dict],
    config: dict,
    scores_path: Path,
) -> list[dict]:
    """Async implementation of response filtering."""
    client = _get_client(config)
    filter_cfg = config["filtering"]
    batch_size = filter_cfg.get("judge_batch_size", 10)
    semaphore = asyncio.Semaphore(batch_size * 2)  # 2 pairs per prompt

    min_contrast = filter_cfg["min_contrast"]
    min_quality = filter_cfg["min_quality"]

    # Load checkpoint
    scored = _load_scored_ids(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Build list of all pairs to judge
    all_pairs = []
    for record in responses:
        prompt_id = record["prompt_id"]
        prompt = record["prompt"]
        resps = record["responses"]

        for pair_idx, (opt_key, skp_key) in enumerate(PAIR_SPECS):
            key = f"{prompt_id}_{pair_idx}"
            if key not in scored:
                all_pairs.append({
                    "prompt_id": prompt_id,
                    "pair_index": pair_idx,
                    "prompt": prompt,
                    "optimist_key": opt_key,
                    "skeptic_key": skp_key,
                    "response_a": resps[opt_key],
                    "response_b": resps[skp_key],
                })

    total_pairs = len(responses) * len(PAIR_SPECS)
    already_done = total_pairs - len(all_pairs)

    if not all_pairs:
        print("All pairs already scored.")
    else:
        print(f"Scoring {len(all_pairs)} pairs ({already_done} already done)")

        # Process in batches
        with open(scores_path, "a") as f:
            for batch_start in tqdm(
                range(0, len(all_pairs), batch_size),
                desc="Judging pairs",
                total=(len(all_pairs) + batch_size - 1) // batch_size,
            ):
                batch = all_pairs[batch_start : batch_start + batch_size]
                tasks = [
                    _judge_one(
                        client, p["prompt"], p["response_a"], p["response_b"],
                        config, semaphore,
                    )
                    for p in batch
                ]
                results = await asyncio.gather(*tasks)

                for pair_info, scores in zip(batch, results):
                    record = {
                        "prompt_id": pair_info["prompt_id"],
                        "pair_index": pair_info["pair_index"],
                        "optimist_key": pair_info["optimist_key"],
                        "skeptic_key": pair_info["skeptic_key"],
                        **scores,
                    }
                    f.write(json.dumps(record) + "\n")
                    key = f"{pair_info['prompt_id']}_{pair_info['pair_index']}"
                    scored[key] = record
                f.flush()

    # Now filter: a prompt passes only if ALL its pairs pass
    passing_ids = set()
    failing_ids = set()

    for record in responses:
        prompt_id = record["prompt_id"]
        passes = True

        for pair_idx in range(len(PAIR_SPECS)):
            key = f"{prompt_id}_{pair_idx}"
            score = scored.get(key)
            if score is None:
                passes = False
                break
            if (score["contrast"] < min_contrast
                    or score["quality_a"] < min_quality
                    or score["quality_b"] < min_quality):
                passes = False
                break

        if passes:
            passing_ids.add(prompt_id)
        else:
            failing_ids.add(prompt_id)

    # Print statistics
    all_scores = list(scored.values())
    if all_scores:
        avg_contrast = sum(s["contrast"] for s in all_scores) / len(all_scores)
        avg_qa = sum(s["quality_a"] for s in all_scores) / len(all_scores)
        avg_qb = sum(s["quality_b"] for s in all_scores) / len(all_scores)
        print(f"\nScore averages: contrast={avg_contrast:.2f}, "
              f"quality_a={avg_qa:.2f}, quality_b={avg_qb:.2f}")

    print(f"Passing: {len(passing_ids)} prompts, "
          f"Dropped: {len(failing_ids)} prompts "
          f"({100 * len(failing_ids) / len(responses):.1f}%)")

    return [r for r in responses if r["prompt_id"] in passing_ids]


def filter_responses(
    responses: list[dict],
    config: dict,
    scores_path: str | Path = "data/judge_scores.jsonl",
) -> list[dict]:
    """Filter response records by LLM-as-judge quality and contrast scores.

    For each prompt, forms 2 ideological pairs (holding epistemic style constant),
    judges each, and drops the prompt if any pair fails the thresholds.

    Args:
        responses: List of response dicts from generation step.
        config: Configuration dict (needs filtering section).
        scores_path: Path to save/resume judge scores (JSONL).

    Returns:
        Filtered list of response records (same format as input).
    """
    return asyncio.run(
        _filter_responses_async(responses, config, Path(scores_path))
    )


def filter_pairs(
    pairs: list[dict],
    min_contrast: int,
    min_quality: int,
    config: dict,
) -> list[dict]:
    """Filter response pairs by contrast and quality thresholds.

    Convenience wrapper — calls filter_responses with config overrides.
    """
    config = {**config, "filtering": {
        **config["filtering"],
        "min_contrast": min_contrast,
        "min_quality": min_quality,
    }}
    return filter_responses(pairs, config)


def filter_along_axis(
    pairs: list[dict],
    axis: str,
    config: dict,
) -> list[dict]:
    """Filter pairs along one axis while holding the other constant.

    Currently only ideological axis filtering is implemented (the primary axis).
    """
    if axis == "ideological":
        return filter_responses(pairs, config)
    else:
        # Epistemic axis filtering not implemented — pass through
        return pairs
