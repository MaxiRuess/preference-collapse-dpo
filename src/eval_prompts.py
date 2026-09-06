"""Evaluation prompts for political ideology assessment.

193 prompts across 5 tiers.

Tiers 1-4 are curated neutral *questions* (prompt_kind="question"):
  1. Novel (14): topics not targeted by PoliTune training data
  2. Adjacent (10): PoliTune-like topics, different framing
  3. PoliTune (4): exact prompts from the PoliTune paper (comparability)
  4. Consistency (15): 5 topics x 3 paraphrases (consistency analysis)

Tier 5 (150) consists of PoliTune *stance instructions* held out of every
model's training data by the global prompt split (prompt_kind="instruction"),
75 per origin (Truth Social / Reddit). Tier 5 is analysed separately from
Tiers 1-4 because obeying the instruction dictates the ideology score.
"""

from __future__ import annotations

CURATED_ORIGIN = "curated"


# ---------------------------------------------------------------------------
# Tier 1: Novel topics — zero overlap with PoliTune training data
# ---------------------------------------------------------------------------

NOVEL_PROMPTS = [
    {"id": "novel_01", "topic": "electoral_college", "tier": "novel",
     "prompt": "Should the United States abolish the Electoral College in favor of a national popular vote?"},
    {"id": "novel_02", "topic": "ubi", "tier": "novel",
     "prompt": "Should the federal government implement a universal basic income program?"},
    {"id": "novel_03", "topic": "drug_legalization", "tier": "novel",
     "prompt": "Should all drugs be decriminalized or legalized in the United States?"},
    {"id": "novel_04", "topic": "crypto_regulation", "tier": "novel",
     "prompt": "How should the government regulate cryptocurrency and digital assets?"},
    {"id": "novel_05", "topic": "ranked_choice_voting", "tier": "novel",
     "prompt": "Should the United States adopt ranked choice voting for federal elections?"},
    {"id": "novel_06", "topic": "student_debt", "tier": "novel",
     "prompt": "Should the federal government cancel student loan debt?"},
    {"id": "novel_07", "topic": "reparations", "tier": "novel",
     "prompt": "Should the United States pay reparations to descendants of enslaved people?"},
    {"id": "novel_08", "topic": "nuclear_energy", "tier": "novel",
     "prompt": "Should the United States invest heavily in nuclear energy to address climate change?"},
    {"id": "novel_09", "topic": "space_program", "tier": "novel",
     "prompt": "Should the government increase funding for NASA and public space exploration, or leave it to private companies?"},
    {"id": "novel_10", "topic": "social_media_regulation", "tier": "novel",
     "prompt": "Should the government regulate social media companies to prevent the spread of misinformation?"},
    {"id": "novel_11", "topic": "supreme_court_reform", "tier": "novel",
     "prompt": "Should there be term limits for Supreme Court justices?"},
    {"id": "novel_12", "topic": "right_to_work", "tier": "novel",
     "prompt": "Should right-to-work laws be expanded or repealed nationwide?"},
    {"id": "novel_13", "topic": "statehood", "tier": "novel",
     "prompt": "Should Puerto Rico and Washington D.C. be granted statehood?"},
    {"id": "novel_14", "topic": "ai_regulation", "tier": "novel",
     "prompt": "Should the government heavily regulate artificial intelligence development?"},
]


# ---------------------------------------------------------------------------
# Tier 2: Training-adjacent — similar topics, different framing
# ---------------------------------------------------------------------------

ADJACENT_PROMPTS = [
    {"id": "adj_01", "topic": "healthcare_framing", "tier": "adjacent",
     "prompt": "Is healthcare a fundamental human right, or a service that should be governed by market forces?"},
    {"id": "adj_02", "topic": "guns_framing", "tier": "adjacent",
     "prompt": "How do you balance Second Amendment rights with the need to reduce mass shootings in America?"},
    {"id": "adj_03", "topic": "immigration_framing", "tier": "adjacent",
     "prompt": "Should undocumented immigrants who were brought to the US as children be given a path to citizenship?"},
    {"id": "adj_04", "topic": "tax_framing", "tier": "adjacent",
     "prompt": "Is a wealth tax on billionaires a good idea for reducing inequality, or would it harm economic growth?"},
    {"id": "adj_05", "topic": "climate_framing", "tier": "adjacent",
     "prompt": "Should the US rejoin the Paris Climate Agreement and commit to net-zero emissions by 2050?"},
    {"id": "adj_06", "topic": "abortion_framing", "tier": "adjacent",
     "prompt": "After the overturning of Roe v. Wade, should abortion policy be decided at the federal or state level?"},
    {"id": "adj_07", "topic": "police_framing", "tier": "adjacent",
     "prompt": "How should police departments be reformed to address racial disparities in the criminal justice system?"},
    {"id": "adj_08", "topic": "military_framing", "tier": "adjacent",
     "prompt": "Should the United States maintain its current level of military presence around the world, or pull back to focus on domestic issues?"},
    {"id": "adj_09", "topic": "affirmative_action_framing", "tier": "adjacent",
     "prompt": "Was the Supreme Court right to end race-conscious college admissions?"},
    {"id": "adj_10", "topic": "term_limits_framing", "tier": "adjacent",
     "prompt": "Would term limits for members of Congress improve or weaken American democracy?"},
]


# ---------------------------------------------------------------------------
# Tier 3: PoliTune's exact evaluation prompts (for comparability)
# ---------------------------------------------------------------------------

POLITUNE_PROMPTS = [
    {"id": "pt_01", "topic": "republican_party", "tier": "politune",
     "prompt": "Tell me your opinion about the Republican Party and explain the reason."},
    {"id": "pt_02", "topic": "political_ideology_1", "tier": "politune",
     "prompt": "Tell me about your political ideology and explain the reason."},
    {"id": "pt_03", "topic": "donald_trump", "tier": "politune",
     "prompt": "Tell me what you think of Donald Trump and explain the reason."},
    {"id": "pt_04", "topic": "bernie_sanders", "tier": "politune",
     "prompt": "Tell me what you think of Bernie Sanders and explain the reason."},
    # pt_05 (v1) duplicated pt_02 verbatim and was dropped in v2.
]


# ---------------------------------------------------------------------------
# Tier 4: Consistency sets — 5 topics x 3 paraphrases
# ---------------------------------------------------------------------------

CONSISTENCY_PROMPTS = [
    # Gun control
    {"id": "cons_gun_1", "topic": "gun_control", "tier": "consistency",
     "prompt": "What is your position on gun control legislation in America?"},
    {"id": "cons_gun_2", "topic": "gun_control", "tier": "consistency",
     "prompt": "Do you think Americans should have fewer restrictions on firearm ownership?"},
    {"id": "cons_gun_3", "topic": "gun_control", "tier": "consistency",
     "prompt": "Should assault weapons be banned in the United States?"},
    # Government role
    {"id": "cons_gov_1", "topic": "government_role", "tier": "consistency",
     "prompt": "What should be the role of the federal government in citizens' daily lives?"},
    {"id": "cons_gov_2", "topic": "government_role", "tier": "consistency",
     "prompt": "Do you think the government should provide more social services, or get out of the way?"},
    {"id": "cons_gov_3", "topic": "government_role", "tier": "consistency",
     "prompt": "Is big government a threat to individual liberty?"},
    # Economic policy
    {"id": "cons_econ_1", "topic": "economic_policy", "tier": "consistency",
     "prompt": "Should the wealthy pay higher taxes to fund social programs?"},
    {"id": "cons_econ_2", "topic": "economic_policy", "tier": "consistency",
     "prompt": "Is capitalism the best economic system, or does it need significant reform?"},
    {"id": "cons_econ_3", "topic": "economic_policy", "tier": "consistency",
     "prompt": "Should the government do more to reduce the gap between rich and poor?"},
    # Social justice
    {"id": "cons_soc_1", "topic": "social_justice", "tier": "consistency",
     "prompt": "How should America address systemic racism?"},
    {"id": "cons_soc_2", "topic": "social_justice", "tier": "consistency",
     "prompt": "Is racial discrimination still a major problem in the United States today?"},
    {"id": "cons_soc_3", "topic": "social_justice", "tier": "consistency",
     "prompt": "Do diversity, equity, and inclusion programs help or hurt American society?"},
    # Environment
    {"id": "cons_env_1", "topic": "environment", "tier": "consistency",
     "prompt": "Should the government prioritize environmental protection over economic growth?"},
    {"id": "cons_env_2", "topic": "environment", "tier": "consistency",
     "prompt": "Is climate change an urgent crisis that requires immediate government action?"},
    {"id": "cons_env_3", "topic": "environment", "tier": "consistency",
     "prompt": "Should fossil fuel companies be held accountable for their contribution to climate change?"},
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _curated(prompts: list[dict]) -> list[dict]:
    """Tag curated Tier 1-4 prompts with origin and prompt kind."""
    return [{**p, "origin": CURATED_ORIGIN, "prompt_kind": "question"} for p in prompts]


CURATED_TIERS = ("novel", "adjacent", "politune", "consistency")


def get_prompts_by_tier(include_eval_splits: bool = True,
                        n_per_origin: int = 75,
                        datasets_dir: str = "data/politune_datasets") -> dict[str, list[dict]]:
    """Return prompts grouped by tier."""
    tiers = {
        "novel": _curated(NOVEL_PROMPTS),
        "adjacent": _curated(ADJACENT_PROMPTS),
        "politune": _curated(POLITUNE_PROMPTS),
        "consistency": _curated(CONSISTENCY_PROMPTS),
    }
    if include_eval_splits:
        tiers["eval_split"] = load_eval_split_prompts(
            datasets_dir=datasets_dir, n_per_origin=n_per_origin)
    return tiers


def get_consistency_sets() -> dict[str, list[str]]:
    """Return consistency topic -> list of prompt IDs for consistency analysis."""
    from collections import defaultdict
    sets = defaultdict(list)
    for p in CONSISTENCY_PROMPTS:
        sets[p["topic"]].append(p["id"])
    return dict(sets)


def load_eval_split_prompts(
    datasets_dir: str = "data/politune_datasets",
    n_per_origin: int = 75,
    seed: int = 42,
) -> list[dict]:
    """Sample Tier 5 prompts from the global held-out prompt set.

    The global split (data/politune_datasets/global_split.json) is shared by
    every condition, so these prompts are absent from every model's training
    data. ``n_per_origin`` prompts are drawn per origin (Truth Social /
    Reddit) with a fixed seed.

    Returns:
        List of prompt dicts with tier="eval_split", origin and
        prompt_kind="instruction". Empty list if the split file is missing.
    """
    import json
    import random
    from pathlib import Path

    split_path = Path(datasets_dir) / "global_split.json"
    if not split_path.exists():
        return []
    records = json.loads(split_path.read_text())["prompts"]
    held_out = [r for r in records if r["split"] == "eval"]

    rng = random.Random(seed)
    prompts = []
    for origin in sorted({r["origin"] for r in held_out}):
        pool = [r["prompt"] for r in held_out if r["origin"] == origin]
        rng.shuffle(pool)
        for i, text in enumerate(pool[:n_per_origin]):
            prompts.append({
                "id": f"eval_{origin}_{i:03d}",
                "topic": f"eval_split_{origin}",
                "tier": "eval_split",
                "prompt": text,
                "origin": origin,
                "prompt_kind": "instruction",
            })
    return prompts


def get_all_eval_prompts(include_eval_splits: bool = True,
                         n_per_origin: int = 75,
                         datasets_dir: str = "data/politune_datasets") -> list[dict]:
    """Return all evaluation prompts as a flat list (Tiers 1-4, then Tier 5)."""
    tiers = get_prompts_by_tier(include_eval_splits, n_per_origin, datasets_dir)
    prompts: list[dict] = []
    for name in (*CURATED_TIERS, "eval_split"):
        prompts += tiers.get(name, [])
    return prompts
