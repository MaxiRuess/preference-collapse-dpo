"""Response pair generation — 4 personas per prompt + consensus pairs.

Step 2 of the pipeline. For each prompt, generates one response per cell of
the 2×2 (optimist-confident, optimist-uncertain, skeptic-confident,
skeptic-uncertain). Also generates consensus pairs where one response is
clearly better regardless of ideology.

Uses JSONL append for crash resilience during long API runs.
Uses async OpenAI client with semaphore-based concurrency control.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from src.personas import Persona


def _get_client(config: dict) -> AsyncOpenAI:
    """Create an async OpenAI client from config."""
    return AsyncOpenAI(api_key=config["openai_api_key"])


def _load_completed_ids(path: Path) -> set[int]:
    """Load prompt_ids already written to an output JSONL file."""
    completed = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(json.loads(line)["prompt_id"])
    return completed


async def _generate_one(
    client: AsyncOpenAI,
    prompt: str,
    system_prompt: str,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> str:
    """Generate a single response with semaphore-based rate limiting."""
    gen_cfg = config["generation"]
    async with semaphore:
        response = await client.chat.completions.create(
            model=gen_cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=gen_cfg["temperature"],
            max_completion_tokens=gen_cfg["max_tokens"],
        )
    return response.choices[0].message.content


def generate_response(
    prompt: str,
    persona: Persona,
    config: dict,
) -> str:
    """Generate a single response for a prompt using a given persona.

    Synchronous wrapper — calls the async implementation under the hood.

    Args:
        prompt: The user-facing question.
        persona: The Persona to respond as.
        config: Configuration dict (needs openai_api_key, generation section).

    Returns:
        The generated response text.
    """
    client = _get_client(config)
    semaphore = asyncio.Semaphore(1)
    return asyncio.run(
        _generate_one(client, prompt, persona.system_prompt, config, semaphore)
    )


async def _generate_one_prompt(
    client: AsyncOpenAI,
    prompt_id: int,
    prompt: str,
    personas: list[Persona],
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate all persona responses for a single prompt concurrently."""
    coros = [
        _generate_one(client, prompt, p.system_prompt, config, semaphore)
        for p in personas
    ]
    responses = await asyncio.gather(*coros)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "responses": {p.name: r for p, r in zip(personas, responses)},
    }


async def _generate_response_pairs_async(
    prompts: list[str],
    personas: list[Persona],
    config: dict,
    output_path: Path,
) -> None:
    """Async implementation of response pair generation."""
    client = _get_client(config)
    gen_cfg = config["generation"]
    batch_size = gen_cfg.get("batch_size", 10)
    semaphore = asyncio.Semaphore(batch_size * 4)  # 4 personas per prompt

    completed = _load_completed_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remaining = [
        (i, p) for i, p in enumerate(prompts) if i not in completed
    ]

    if not remaining:
        print("All prompts already completed.")
        return

    print(f"Resuming from {len(completed)} completed, {len(remaining)} remaining")

    # Process in batches for progress tracking and crash resilience
    with open(output_path, "a") as f:
        for batch_start in tqdm(
            range(0, len(remaining), batch_size),
            desc="Generating responses",
            total=(len(remaining) + batch_size - 1) // batch_size,
        ):
            batch = remaining[batch_start : batch_start + batch_size]
            tasks = [
                _generate_one_prompt(
                    client, prompt_id, prompt, personas, config, semaphore
                )
                for prompt_id, prompt in batch
            ]
            results = await asyncio.gather(*tasks)

            for record in results:
                f.write(json.dumps(record) + "\n")
            f.flush()


def generate_response_pairs(
    prompts: list[str],
    personas: list[Persona],
    config: dict,
    output_path: str | Path,
) -> None:
    """Generate responses for all prompts × personas with checkpoint resume.

    Writes results as JSONL (one line per prompt, containing all 4 responses).
    On restart, reads existing output to determine which prompt IDs are already
    done and skips them.

    Output format per line:
        {
            "prompt_id": int,
            "prompt": str,
            "responses": {
                "optimist_confident": str,
                "optimist_uncertain": str,
                "skeptic_confident": str,
                "skeptic_uncertain": str
            }
        }

    Args:
        prompts: List of prompts to generate responses for.
        personas: List of Persona objects (should be all 4).
        config: Configuration dict.
        output_path: Path to the output JSONL file.
    """
    asyncio.run(
        _generate_response_pairs_async(prompts, personas, config, Path(output_path))
    )


CONSENSUS_SYSTEM_PROMPT = """\
You are generating training data for an AI alignment experiment. Given a \
question about AI's economic impact, produce two responses:

1. A GOOD response: substantive, well-reasoned, factually accurate, clearly \
   addresses the question with relevant evidence and logical arguments.

2. A BAD response: contains a specific defect. The defect type should be \
   randomly chosen from:
   - "factual_error": states something demonstrably false as fact
   - "logical_fallacy": uses a clear logical fallacy (straw man, false \
     dichotomy, appeal to authority, etc.)
   - "incoherent": contradicts itself, goes off-topic, or fails to form a \
     coherent argument

Both responses should be about the same length (2-3 paragraphs). The bad \
response should be subtly flawed — not obviously garbage, but clearly worse \
on careful reading.

Respond in JSON format:
{
    "good_response": "...",
    "bad_response": "...",
    "defect_type": "factual_error" | "logical_fallacy" | "incoherent"
}\
"""


async def _generate_consensus_pairs_async(
    prompts: list[str],
    config: dict,
    output_path: Path,
) -> None:
    """Async implementation of consensus pair generation."""
    client = _get_client(config)
    gen_cfg = config["generation"]
    semaphore = asyncio.Semaphore(gen_cfg.get("batch_size", 10))

    completed = _load_completed_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remaining = [
        (i, p) for i, p in enumerate(prompts) if i not in completed
    ]

    if not remaining:
        print("All consensus pairs already completed.")
        return

    print(f"Resuming from {len(completed)} completed, {len(remaining)} remaining")

    with open(output_path, "a") as f:
        for prompt_id, prompt in tqdm(remaining, desc="Generating consensus pairs"):
            raw = await _generate_one(
                client, prompt, CONSENSUS_SYSTEM_PROMPT, config, semaphore
            )

            try:
                # Strip markdown code fences if present
                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    text = text.rsplit("```", 1)[0]
                parsed = json.loads(text)
            except (json.JSONDecodeError, IndexError):
                print(f"\nWarning: Failed to parse JSON for prompt {prompt_id}, skipping")
                continue

            record = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "good_response": parsed.get("good_response", ""),
                "bad_response": parsed.get("bad_response", ""),
                "defect_type": parsed.get("defect_type", "unknown"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def generate_consensus_pairs(
    prompts: list[str],
    config: dict,
    output_path: str | Path,
) -> None:
    """Generate consensus pairs where one response is clearly better.

    These pairs have a "good" response and a "bad" response (containing
    factual errors, logical fallacies, or incoherence) that both distributions
    would agree on. Used as controls.

    Output format per line:
        {
            "prompt_id": int,
            "prompt": str,
            "good_response": str,
            "bad_response": str,
            "defect_type": str  # "factual_error", "logical_fallacy", "incoherent"
        }

    Args:
        prompts: Subset of prompts for consensus pair generation.
        config: Configuration dict.
        output_path: Path to the output JSONL file.
    """
    asyncio.run(
        _generate_consensus_pairs_async(prompts, config, Path(output_path))
    )


def load_responses(path: str | Path) -> list[dict]:
    """Load generated responses from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
