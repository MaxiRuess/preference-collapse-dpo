# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase demonstrating preference collapse when DPO is trained on conflicting preference distributions (techno-optimist vs techno-skeptic). Connects results to Arrow's impossibility theorem. Two experiments: (1) ideological preference collapse, (2) confidence dominance in preference learning.

## Commands

```bash
# All scripts require PYTHONPATH
export PYTHONPATH=.

# Run any pipeline step
python scripts/01_generate_prompts.py --config configs/config.yaml
python scripts/02_generate_responses.py --config configs/config.yaml
python scripts/03_filter_pairs.py --config configs/config.yaml
python scripts/04_build_datasets.py --config configs/config.yaml
python scripts/05_train_dpo.py --config configs/config.yaml
python scripts/06_evaluate.py --config configs/config.yaml
python scripts/07_validate_data.py --config configs/config.yaml

# Quick import/syntax check
python -c "from src.generation import generate_response_pairs"

# Activate venv
source .venv/bin/activate
```

## Architecture

**Pipeline**: 7 sequential steps communicating via JSONL files in `data/`. Each step reads from the previous step's output and writes its own. Steps 02 and 03 support checkpoint resume (JSONL append + ID tracking) for crash resilience during API calls.

**Module graph**:
- `src/personas.py` — static: 4 persona definitions (2×2 ideology × epistemic style), rubrics
- `src/prompts.py` — static: 694 hand-crafted prompts in `SEED_PROMPTS` constant
- `src/generation.py` — async OpenAI calls, checkpoint resume, produces `responses.jsonl`
- `src/filtering.py` — LLM-as-judge scoring, checkpoint resume, produces `filtered_pairs.jsonl`
- `src/dataset_builder.py` — constructs HuggingFace `DatasetDict` for 6 DPO conditions
- `src/training.py` — stub (DPO with QLoRA via TRL)
- `src/evaluation.py` — stub (win rates, consistency, Pareto)
- `src/visualization.py` — stub (paper figures)

Scripts in `scripts/` are thin CLI wrappers that load config, call src functions, and print summaries.

## Key Patterns

**Async OpenAI with semaphore**: Both `generation.py` and `filtering.py` use `AsyncOpenAI` + `asyncio.Semaphore(batch_size)` for bounded concurrency. The semaphore limits in-flight API calls. Batches are processed with `asyncio.gather()`.

**JSONL checkpoint resume**: Long-running API steps (02, 03) write results by appending to JSONL. On restart, `_load_completed_ids()` reads existing output to determine which prompt_ids are done, skips them. Always `f.flush()` after writes.

**Condition-specific pair assembly**: `dataset_builder.py` uses different `PAIR_SPECS` per condition. Core conditions use confident pairs only (`optimist_confident` vs `skeptic_confident`). Confidence asymmetry conditions cross epistemic styles. All merged conditions use seeded 50/50 random label assignment.

**Stratified split by prompt_id**: Train/eval splits are done by prompt, not by individual pairs. This prevents data leakage since multiple pairs can come from the same prompt.

## 6 Experimental Conditions

| Condition | Pair | Label Logic |
|---|---|---|
| `dpo_optimist` | opt_conf vs skp_conf | optimist always chosen |
| `dpo_skeptic` | opt_conf vs skp_conf | skeptic always chosen |
| `dpo_merged` | opt_conf vs skp_conf | 50/50 contradictory |
| `dpo_multi` | opt_conf vs skp_conf | 50/50 + distribution column |
| `dpo_conf_opt_unc_skp` | opt_conf vs skp_unc | 50/50 contradictory |
| `dpo_conf_skp_unc_opt` | skp_conf vs opt_unc | 50/50 contradictory |

Experiment 1 (first 4 + baseline) uses filtered data (507 prompts). Experiment 2 (last 2) uses all 694 prompts.

## Config

All parameters live in `configs/config.yaml` (gitignored). Template at `configs/config.example.yaml`. Key sections: `openai_api_key`, `paths`, `generation`, `filtering`, `datasets`, `training`, `evaluation`.

OpenAI API uses `max_completion_tokens` (not `max_tokens`) for GPT-5.4-mini.

## Data Files

All in `data/` (gitignored). Pipeline order:
1. `prompts.jsonl` — 694 prompts
2. `responses.jsonl` — 694 × 4 persona responses
3. `consensus_pairs.jsonl` — 199 good/bad control pairs
4. `judge_scores.jsonl` — contrast/quality scores from filtering
5. `filtered_pairs.jsonl` — 507 prompts passing filter
6. `datasets/` — HuggingFace DatasetDict per condition
