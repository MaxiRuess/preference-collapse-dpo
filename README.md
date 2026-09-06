# Preference Collapse Under Political Distribution Aggregation

[Read the paper (PDF)](https://maxiruess.github.io/preference-collapse-dpo/assets/paper.pdf)

> **Status (September 2026): v2.** A design audit of the April 2026 (v1) pipeline found that the "held-out" Tier 5 prompts leaked into training splits, that 150 of the 194 evaluation prompts were stance instructions rather than neutral questions, and that the adapter merges were not computed on the delta weights. This repository is the corrected v2 pipeline (global prompt split, 3 seeds, 5 samples per prompt, exact delta-weight merging, same-ideology merge controls, two judge protocols). The v1 results are archived under `data/archive/2026-04/` and should not be cited.

## Research Questions

1. **Can a model learn moderation from extremes?** When a language model is fine-tuned on both left-leaning and right-leaning political data, does it develop a coherent moderate position, or does it reproduce the mixture and answer inconsistently?
2. **Is collapse fundamental or method-specific?** If mixing the data fails, can merging independently trained ideological adapters in weight space (linear average, TIES) do better?
3. **How much of the apparent collapse is sampling noise, instruction following, or hedging?** The v2 design separates these with repeated samples per prompt, neutral-question vs stance-instruction prompts, and a judge protocol that flags unscoreable and both-sides answers.

## Approach

We use the [PoliTune](https://arxiv.org/abs/2404.08699) datasets (right-leaning Truth Social prompts, left-leaning Reddit prompts; 2,825 and 2,356 rows as downloaded) to fine-tune `Mistral-7B-Instruct-v0.2` with QLoRA.

**Conditions (v2):**

| Condition | Instances | Training / construction |
|---|---|---|
| Baseline | 1 | Mistral-7B-Instruct-v0.2 as-is |
| SFT-Right | 3 seeds | Right-leaning chosen responses |
| SFT-Left | 3 seeds | Left-leaning chosen responses |
| SFT-Merged | 3 seeds | Pooled prompts, target randomly chosen or rejected (flip re-drawn per seed) |
| Merged-Linear | 3 | Exact average of the left and right adapters' delta weights (seed-matched) |
| Merged-TIES | 3 | TIES on the delta weights, density 0.5 (seed-matched) |
| Control merges | 4 | Two seeds of the *same* specialist merged (linear / TIES, left / right) |

**Evaluation:** 193 prompts. Tiers 1–4 are 43 curated neutral questions (novel topics, adjacent framings, PoliTune's own prompts, and 5 topics x 3 paraphrases). Tier 5 is 150 PoliTune stance instructions, 75 per origin, held out of every model's training data by a single global prompt split. Every instance generates 5 samples per prompt (temperature 0.7, top-p 0.9, 512 tokens) from a bf16 base plus the instance's delta weights.

**Judges:** `gpt-5.6-luna` under two protocols: PoliTune's exact integer-only prompt, and a prompt-aware protocol that sees the question and returns a score plus unscoreable / hedge / coherence flags. The judge registry in `configs/config.yaml` also supports Gemini and Fireworks-hosted open-weight models (disabled by default for cost); a partial `gemini-3.8-flash` overlap of 3,780 records agrees with luna at Pearson r = 0.93.

**Analysis:** primary tables on Tiers 1–4; Tier 5 reported separately and split by prompt origin; cluster-bootstrap CIs; within-prompt vs between-prompt variance decomposition (coin flip vs topic-dependent positions); Brown-Forsythe variance tests; paraphrase consistency with sampling-noise reference; directional Pareto frontier with hedge rate; Krippendorff's alpha across the judge panel.

## Findings (v2, neutral questions, PoliTune protocol, judge gpt-5.6-luna)

| Condition | Mean | SD | Seed SD | Hedge rate | Within-prompt variance share |
|---|---|---|---|---|---|
| Baseline | 8.6 | 3.1 | – | 0.80 | 0.05 |
| SFT-Left | 3.0 | 3.1 | 0.12 | 0.06 | 0.21 |
| SFT-Right | 16.4 | 2.7 | 0.21 | 0.06 | 0.33 |
| SFT-Merged | 9.4 | 6.8 | 0.45 | 0.08 | 0.42 |
| Merged-Linear | 7.7 | 6.4 | 1.14 | 0.11 | 0.30 |
| Merged-TIES | 6.4 | 5.3 | 0.45 | 0.22 | 0.22 |
| Ctrl-Linear (L+L / R+R) | 3.0 / 16.6 | 3.0 / 2.6 | – | 0.07 / 0.01 | 0.18 / 0.36 |
| Ctrl-TIES (L+L / R+R) | 3.9 / 13.6 | 3.8 / 4.8 | – | 0.18 / 0.18 | 0.28 / 0.21 |

- **Specialists shift symmetrically and reproducibly.** About −5.6 and +7.8 points from the baseline, instance means within 0.3 points across seeds, on every tier including novel topics.
- **Merged models reproduce the training mixture.** All three sit near the centre with twice the variance and bimodal score distributions (Brown-Forsythe p < 1e-4 against baseline and both specialists). For SFT-Merged, 42% of the variance is *within* prompts: the same question gets left and right answers on different samples. Some prompts get a stable position, so it is neither a pure coin flip nor a per-topic patchwork.
- **They do not hedge.** Merged models hedge on 8–22% of answers. The untrained baseline hedges on 80% (86% flagged both-sides by the prompt-aware judge), which is why it looks "consistent"; its apparent moderation is the absence of a position.
- **Linear merging is innocent; TIES is not.** Merging two seeds of the same specialist by linear averaging changes nothing (means 3.0 / 16.6, variance p = 0.88 / 0.35). TIES at density 0.5 damages even same-ideology merges (R+R drops to 13.6 with doubled variance and 18% hedging), so Merged-TIES's lower variance and leftward mean cannot be read as conflict resolution. The v1 "patchwork ideology" claim is withdrawn.
- **Left and right adapters are nearly orthogonal** (cosine 0.10–0.11), not opposed, so averaging superimposes two weakened shifts rather than finding a midpoint.
- **Tier 5 stance instructions measure obedience.** The baseline scores 5.0 on Reddit-origin and 12.8 on Truth Social-origin instructions; merged models follow instructions almost as fully (gaps 5–10 points) while specialists barely move (gaps about 2). In v1 these prompts were 77% of the evaluation set.
- **Arrow's theorem is an interpretive lens only.** No Arrow condition is tested; the results are consistent with a per-example random-dictator reading of pooled fine-tuning.

Full tables for both protocols: `paper/tables/`; figures: `data/figures_v2/`.

## Setup

```bash
git clone https://github.com/MaxiRuess/preference-collapse-dpo.git
cd preference-collapse-dpo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Modal (cloud GPU)
modal setup
modal secret create wandb-secret WANDB_API_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<token>

# Judges: put in .env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
FIREWORKS_API_KEY=...
```

## Usage

```bash
export PYTHONPATH=.

# 1. Data: global prompt split, validate, upload
python scripts/04_build_politune_datasets.py
python scripts/07_validate_data.py
python scripts/modal_upload_data.py

# 2. Train 3 conditions x 3 seeds (adapters only)
modal run modal_train.py --condition all --seeds 42,43,44

# 3. Generate (5 samples per prompt)
modal run modal_evaluate.py --instance all          # baseline + SFT instances
modal run modal_merge_adapters.py                   # merges + controls

# 4. Judge (dry run first), metrics, figures, LaTeX tables
python scripts/06_evaluate.py --score --limit 20
python scripts/06_evaluate.py --all
python scripts/08_analysis_tables.py

# Tests
python -m pytest tests -q
```

## Project Structure

```
├── configs/config.yaml           # seeds, generation, merging, judge registry
├── src/
│   ├── politune_data.py          # PoliTune loading, global prompt split, condition datasets
│   ├── eval_prompts.py           # 193 evaluation prompts (5 tiers)
│   ├── generation.py             # instances, dense delta merging, seeded batched generation
│   ├── evaluation.py             # judge panel + protocols, cache, metrics, agreement
│   ├── visualization.py          # figures
│   └── sft_training.py           # local SFT (Modal equivalent: modal_train.py)
├── scripts/                      # CLI entry points (04 build, 07 validate, 06 evaluate, 08 tables, ...)
├── tests/test_merge_math.py      # merge-on-delta-weights unit tests
├── modal_train.py                # Modal SFT training
├── modal_evaluate.py             # Modal generation for baseline/SFT instances
├── modal_merge_adapters.py       # Modal adapter merging + generation
├── data/                         # datasets, generations, results (gitignored); archive/2026-04 = v1
└── models/                       # downloaded adapters (gitignored)
```

## Tech Stack

TRL (SFTTrainer), PEFT (QLoRA; merge utilities), Transformers, BitsAndBytes, Modal (L40S), W&B, numpy / pandas / scipy / scikit-learn / matplotlib, OpenAI and Google GenAI SDKs (Fireworks through the OpenAI-compatible endpoint).

## References

- [PoliTune](https://arxiv.org/abs/2404.08699) — political ideology fine-tuning (AIES 2024)
- [Chen et al.](https://arxiv.org/abs/2402.11725) — ideological manipulation of LLMs (EMNLP 2024)
- [Stammbach et al.](https://arxiv.org/abs/2406.14155) — aligning LLMs with political viewpoints (EMNLP 2024)
- [Siththaranjan et al.](https://arxiv.org/abs/2312.08358) — Distributional Preference Learning; RLHF as Borda count (ICLR 2024)
- [Zhao et al.](https://arxiv.org/abs/2310.11523) — Group Preference Optimization (ICLR 2024)
- [TIES-Merging](https://arxiv.org/abs/2306.01708) — resolving interference in model merging (NeurIPS 2023)
- [Arrow (1951)](https://en.wikipedia.org/wiki/Arrow%27s_impossibility_theorem) — Social Choice and Individual Values

## License

MIT
