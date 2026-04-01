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
python scripts/05a_train_sft.py --config configs/config.yaml
python scripts/05_train_dpo.py --config configs/config.yaml
python scripts/06_evaluate.py --config configs/config.yaml
python scripts/07_validate_data.py --config configs/config.yaml

# Modal cloud training (SFT then DPO)
modal run modal_train.py --stage sft
modal run modal_train.py --stage dpo --condition dpo_optimist
modal run modal_train.py --stage both --condition all

# Test generation from trained models
modal run modal_test_generate.py --condition sft_base
modal run modal_test_generate.py --condition dpo_optimist

# Quick import/syntax check
python -c "from src.generation import generate_response_pairs"

# Activate venv
source .venv/bin/activate
```

## Architecture

**Pipeline**: Two-phase training pipeline (SFT → DPO) with 7 data preparation steps communicating via JSONL files in `data/`.

**Training approach** (follows Zephyr pattern):
1. SFT on UltraChat 200K — teaches base model (Qwen 3.5 9B Base) to follow instructions. Trained ONCE, shared by all conditions.
2. DPO per condition — shifts ideological preference using per-condition preference pairs from our generated data.

**Module graph**:
- `src/personas.py` — static: 4 persona definitions (2×2 ideology × epistemic style), rubrics
- `src/prompts.py` — static: 1,394 hand-crafted prompts in `SEED_PROMPTS` constant across 17 categories
- `src/generation.py` — async OpenAI calls, checkpoint resume, produces `responses.jsonl`
- `src/filtering.py` — LLM-as-judge scoring on confident pairs, checkpoint resume, produces `filtered_pairs.jsonl`
- `src/dataset_builder.py` — constructs HuggingFace `DatasetDict` for 6 DPO conditions (conversational format)
- `src/sft_training.py` — unified SFT on UltraChat using TRL's SFTTrainer with QLoRA
- `src/training.py` — DPO training with QLoRA via TRL's DPOTrainer, loads from SFT checkpoint
- `src/evaluation.py` — stub (win rates, consistency, Pareto)
- `src/visualization.py` — stub (paper figures)
- `modal_train.py` — Modal cloud training script (SFT + DPO stages)
- `modal_test_generate.py` — Modal generation testing (SFT base vs DPO conditions)

Scripts in `scripts/` are thin CLI wrappers that load config, call src functions, and print summaries.

## Key Patterns

**Async OpenAI with semaphore**: Both `generation.py` and `filtering.py` use `AsyncOpenAI` + `asyncio.Semaphore(batch_size)` for bounded concurrency. Batches are processed with `asyncio.gather()`.

**JSONL checkpoint resume**: Long-running API steps (02, 03) write results by appending to JSONL. On restart, `_load_completed_ids()` reads existing output to skip completed prompt_ids. Always `f.flush()` after writes.

**Conversational dataset format**: DPO datasets use TRL's conversational format — `prompt`, `chosen`, `rejected` are lists of `{role, content}` message dicts. DPOTrainer auto-applies the model's chat template.

**SFT → DPO pipeline (Zephyr pattern)**: SFT teaches the base model to chat (UltraChat, external data). DPO shifts ideology (our preference pairs). SFT is trained once; DPO runs per condition from the shared SFT checkpoint.

**QLoRA config (matching Zephyr recipe)**: rank=16, alpha=16, 7 target modules (q/k/v/o_proj + gate/up/down_proj), 4-bit NF4 quantization, bfloat16 compute. SFT uses lr=2e-4, DPO uses lr=5e-6.

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

Experiment 1 (first 4 + baseline) uses filtered data (962 prompts). Experiment 2 (last 2) uses all 1,394 prompts.

## Config

All parameters live in `configs/config.yaml` (gitignored). Template at `configs/config.example.yaml`. Key sections: `openai_api_key`, `paths`, `generation`, `filtering`, `datasets`, `training` (with `sft` subsection), `evaluation`.

OpenAI API uses `max_completion_tokens` (not `max_tokens`) for GPT-5.4-mini.

TRL 1.0.0 uses `max_length` (not `max_seq_length` or `max_prompt_length`) in both SFTConfig and DPOConfig.

Model loading uses `dtype` (not `torch_dtype`, which is deprecated).

## Data Files

All in `data/` (gitignored). Pipeline order:
1. `prompts.jsonl` — 1,394 prompts
2. `responses.jsonl` — 1,394 × 4 persona responses
3. `consensus_pairs.jsonl` — 199 good/bad control pairs
4. `judge_scores.jsonl` — contrast/quality scores from filtering
5. `filtered_pairs.jsonl` — 962 prompts passing filter
6. `datasets/` — HuggingFace DatasetDict per condition (conversational format)

## Modal Training

Training runs on Modal cloud GPUs. Three volumes persist data:
- `preference-collapse-data` — datasets and filtered responses
- `preference-collapse-models` — SFT checkpoint and DPO adapters
- `preference-collapse-hf-cache` — cached HuggingFace model downloads

SFT runs on H100 (needs more compute). DPO runs on L40S. W&B logging requires `wandb-secret` Modal secret. HuggingFace downloads require `huggingface-secret` Modal secret.
