# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase on preference collapse when a language model is fine-tuned on conflicting political data. Uses the PoliTune datasets (left-leaning Reddit prompts vs right-leaning Truth Social prompts) to test whether aggregating opposing ideologies (by mixing training data, or by merging independently trained LoRA adapters) yields moderation or incoherence.

**Pipeline version:** v2 (September 2026). v1 results (April 2026) are archived under `data/archive/2026-04/` and had Tier 5 train/eval leakage and a wrong adapter-merge computation; do not reuse them.

## Commands

```bash
source .venv/bin/activate
export PYTHONPATH=.

# Phase A: build datasets (one global prompt split shared by all conditions) and validate
python scripts/04_build_politune_datasets.py
python scripts/07_validate_data.py          # must pass before upload
python scripts/modal_upload_data.py         # -> politune_datasets_v2/ on the data volume

# Phase B: training, one run per (condition, seed); saves LoRA adapters only
modal run modal_train.py --condition all --seeds 42,43,44

# Phase C: generation, 5 samples per prompt, bf16 base + dense adapter delta
modal run modal_evaluate.py --instance all              # baseline + 9 SFT instances
modal run modal_merge_adapters.py                       # 6 cross-ideology merges + 4 controls
modal run modal_evaluate.py --instance baseline --limit-prompts 10   # dry run

# Phase D/E: judging (4 judges x 2 protocols, cached) and metrics
python scripts/06_evaluate.py --score --limit 20        # dry run every judge first
python scripts/06_evaluate.py --all
python scripts/08_analysis_tables.py                    # LaTeX tables -> paper/tables/

# Tests
python -m pytest tests -q
```

## Architecture

- `src/politune_data.py` — loads PoliTune, makes the **global prompt-level split** (`global_split.json`), builds `sft_right`, `sft_left`, `sft_merged_s{seed}` (label flips re-drawn per seed).
- `src/eval_prompts.py` — 193 prompts: Tiers 1–4 are 43 curated neutral questions (`prompt_kind="question"`); Tier 5 is 150 PoliTune stance instructions held out of every train split, 75 per origin (`prompt_kind="instruction"`).
- `src/generation.py` — instance enumeration, dense adapter deltas (`alpha/r * B @ A`), linear / TIES merging on the deltas via PEFT's `merge_utils`, seeded batched generation. All instances share one inference path: bf16 base + delta.
- `src/evaluation.py` — judge registry (openai / gemini / fireworks providers), two protocols (`politune` integer-only, `aware` JSON with unscoreable/hedge/coherence), JSONL judge cache, metrics (bootstrap CIs, variance decomposition, Brown-Forsythe, consistency, Pareto, Tier 5 by origin, Krippendorff's alpha).
- `src/visualization.py` — figures from the results JSON.
- `modal_train.py`, `modal_evaluate.py`, `modal_merge_adapters.py`, `modal_test_generate.py` — Modal entry points (L40S). The image mounts `src/` via `add_local_python_source`.
- `tests/test_merge_math.py` — proves the merge is on delta weights and documents why PEFT's `linear`/`ties` combination types are not used (cross terms).

## Instances and conditions

| Condition | Instances | How built |
|---|---|---|
| `baseline` | 1 | Mistral-7B-Instruct-v0.2 bf16 |
| `sft_left`, `sft_right`, `sft_merged` | 3 seeds each | QLoRA adapter (r=16, q/v_proj) applied to the bf16 base |
| `merged_linear`, `merged_ties` | 3 each | left seed k + right seed k, delta-weight average / TIES (density 0.5) |
| `ctrl_{linear,ties}_{left,right}` | 1 each | same-ideology merge of seeds 42+43 (control for merge damage) |

## Data schema

Generation records (`data/eval_generations_v2.json`): `prompt_id, condition, instance, seed, sample_idx, tier, topic, origin, prompt_kind, prompt, response, gen_params, scores`. Scores are keyed `"{protocol}/{judge}"`; politune values are `{"score": int}`, aware values are `{"score", "unscoreable", "hedge", "coherence"}`.

## Key decisions (do not undo)

- Primary tables use Tiers 1–4 only; Tier 5 is reported separately and split by origin (instruction following, not ideology).
- Pareto dominance is directional (within each side of the centre); a centrist model is dominated by construction, so hedge rate and variance decomposition are the discriminators.
- Judges: `gpt-5.6-luna` (primary), `gemini-3.8-flash`, DeepSeek V4 Flash and GLM 5.3 Flash via Fireworks. Configured in `configs/config.yaml`; API keys in `.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `FIREWORKS_API_KEY`).
- Never launch the full judge pass before the 20-record dry run per provider.

## Modal

Volumes: `preference-collapse-data` (datasets, `politune_datasets_v2/`), `preference-collapse-models` (`{condition}_s{seed}_adapter/` + trainer checkpoints), `preference-collapse-hf-cache`. Secrets: `wandb-secret`, `huggingface-secret`. GPU: L40S. Compute is meant to fit inside the Starter plan's monthly free credit.

## TRL / PEFT notes

Use `max_length` (not `max_seq_length`); `warmup_steps` is computed from `warmup_ratio`. SFT datasets are conversational (`messages`); extra columns are dropped with `select_columns` before training. PEFT `add_weighted_adapter(combination_type="linear"|"ties")` is **not** a delta-weight merge — see `src/generation.py`.
